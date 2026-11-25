//! MoE expert offload, the disk backing's miss path: how a launch's routed
//! experts reach their VRAM slots when the full planes live in an artifact
//! on disk (`expert_store.rs`) instead of device-mapped host memory.
//!
//! The pinned path resolves and fills on the device, inside the graphs
//! (`moe_cache.rs`). A disk read cannot run there, so this path resolves on
//! the host: the routed ids come down, a host LRU over the layer's slots
//! gives each distinct expert a slot, the misses are read into staging
//! buffers and copied into their slots in stream order, and the slot routing
//! goes back up. The LRU walk is the resolve kernel's (`moe/offload.cuh`:
//! same victim choice), so both backings see the same hits and misses.
//!
//! A slot whose fill did not complete is never consumed. A failed read or
//! enqueue invalidates every slot the launch assigned before the error
//! returns. A failed wait breaks the backing for good: the device may still
//! be reading the staging buffers, so they are never handed out again.
//!
//! The device is behind [`SlotCopier`] (enqueue a host-to-slot copy, wait for
//! every enqueued copy); the tests drive it with a recording fake.

use std::ops::Range;
use std::time::{Duration, Instant};

use super::GpuError;
use super::expert_store::{DiskExpertStore, StagingLease, StagingPool};
use super::moe_cache::MOE_CACHE_NONE;

/// The device half of a fill: copies from host staging memory into one
/// layer's slot planes, on the stream the expert kernels run on.
pub(super) trait SlotCopier {
    /// Enqueue a copy of `src` into stream `stream` (fill-descriptor order:
    /// gate data, gate scales, up data, up scales, down data, down scales)
    /// of `slot`.
    ///
    /// # Safety
    /// `src` must stay allocated and unmodified until the next [`wait`]
    /// returns `Ok`; the copy may still be reading it until then.
    ///
    /// [`wait`]: SlotCopier::wait
    unsafe fn enqueue(&mut self, stream: usize, slot: u32, src: &[u8]) -> Result<(), GpuError>;

    /// Block until every enqueued copy has completed.
    fn wait(&mut self) -> Result<(), GpuError>;
}

/// Counters for the disk backing since load, summed over layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoeDiskStats {
    /// Rows resolved (routed `(token, expert)` pairs), as the pinned
    /// cache's `stats()` counts them.
    pub rows: u64,
    /// Experts read from disk.
    pub misses: u64,
    pub bytes_read: u64,
    pub read_time: Duration,
    /// Times a fill found every staging buffer in flight and waited.
    pub staging_waits: u64,
    pub wait_time: Duration,
}

/// Host-side LRU over one layer's slots: the state the resolve kernel keeps
/// in device memory on the pinned path, with the same walk.
struct SlotLru {
    slot_of: Vec<u32>,
    expert_in: Vec<u32>,
    last_use: Vec<u32>,
    tick: u32,
}

/// One miss: read `expert` into `slot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MissJob {
    slot: u32,
    expert: u32,
}

impl SlotLru {
    fn new(n_expert: usize, slots: usize) -> Self {
        Self {
            slot_of: vec![MOE_CACHE_NONE; n_expert],
            expert_in: vec![MOE_CACHE_NONE; slots],
            last_use: vec![0; slots],
            tick: 0,
        }
    }

    /// Resolve `rows` routed ids: each row's slot into `slot_rows`, each
    /// miss appended to `jobs`. The caller guarantees the rows' distinct
    /// experts fit the slots (`plan_chunks`), so a tick never evicts an
    /// expert it reads; the slots in `jobs` hold nothing valid until their
    /// fill completes.
    fn resolve(&mut self, rows: &[u32], slot_rows: &mut [u32], jobs: &mut Vec<MissJob>) {
        self.tick += 1;
        let t = self.tick;
        for (r, &e) in rows.iter().enumerate() {
            let s = self.slot_of[e as usize];
            if s != MOE_CACHE_NONE {
                slot_rows[r] = s;
                self.last_use[s as usize] = t;
                continue;
            }
            let (victim, _) = self
                .last_use
                .iter()
                .enumerate()
                .filter(|&(_, &lu)| lu != t)
                .min_by_key(|&(c, &lu)| (lu, c))
                .expect("plan_chunks keeps a tick's distinct experts within the slots");
            let old = self.expert_in[victim];
            if old != MOE_CACHE_NONE {
                self.slot_of[old as usize] = MOE_CACHE_NONE;
            }
            self.expert_in[victim] = e;
            self.slot_of[e as usize] = victim as u32;
            self.last_use[victim] = t;
            slot_rows[r] = victim as u32;
            jobs.push(MissJob {
                slot: victim as u32,
                expert: e,
            });
        }
    }

    /// Forget what `jobs` assigned: the slots read as empty and are taken
    /// first, the experts as absent.
    fn invalidate(&mut self, jobs: &[MissJob]) {
        for j in jobs {
            if self.expert_in[j.slot as usize] == j.expert {
                self.expert_in[j.slot as usize] = MOE_CACHE_NONE;
                self.slot_of[j.expert as usize] = MOE_CACHE_NONE;
                self.last_use[j.slot as usize] = 0;
            }
        }
    }
}

/// Split a launch's routing (`n_active` rows per token) into consecutive
/// token ranges whose distinct experts fit `slots`, greedily. Every range is
/// served by one resolve, so a range never evicts what it reads.
pub(super) fn plan_chunks(
    ids: &[u32],
    n_active: usize,
    n_expert: usize,
    slots: usize,
    out: &mut Vec<Range<usize>>,
) -> Result<(), GpuError> {
    out.clear();
    if n_active == 0 || slots < n_active || !ids.len().is_multiple_of(n_active) {
        return Err(GpuError::Driver(format!(
            "expert disk backing: {} routed rows at top-{n_active} over {slots} slots",
            ids.len()
        )));
    }
    if let Some(&e) = ids.iter().find(|&&e| e as usize >= n_expert) {
        return Err(GpuError::Driver(format!(
            "expert disk backing: routed expert {e} over {n_expert}"
        )));
    }
    // chunk-stamped membership: no clear between chunks
    let mut seen = vec![0u32; n_expert];
    let (mut stamp, mut start, mut distinct) = (1u32, 0usize, 0usize);
    for (t, tok) in ids.chunks(n_active).enumerate() {
        let new = tok
            .iter()
            .enumerate()
            .filter(|&(i, &e)| seen[e as usize] != stamp && !tok[..i].contains(&e))
            .count();
        if distinct + new > slots {
            out.push(start..t);
            stamp += 1;
            start = t;
            distinct = 0;
        }
        for &e in tok {
            if seen[e as usize] != stamp {
                seen[e as usize] = stamp;
                distinct += 1;
            }
        }
    }
    out.push(start..ids.len() / n_active);
    Ok(())
}

/// The disk backing of one model: the artifact, the staging pool every layer
/// shares (layers run one at a time), and each MoE layer's slot LRU.
pub(super) struct DiskExperts<B> {
    store: DiskExpertStore,
    pool: StagingPool<B>,
    lru: Vec<SlotLru>,
    /// Leases whose copies were enqueued and not yet waited for.
    in_flight: Vec<StagingLease>,
    jobs: Vec<MissJob>,
    broken: bool,
    stats: MoeDiskStats,
}

impl<B: AsRef<[u8]> + AsMut<[u8]>> DiskExperts<B> {
    /// `slots` per layer over every layer of `store`. The pool's buffers must
    /// each hold one record of the largest layer.
    pub(super) fn new(
        store: DiskExpertStore,
        pool: StagingPool<B>,
        slots: usize,
    ) -> Result<Self, GpuError> {
        let n_layers = store.n_layers();
        let mut lru = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let layout = store.layout(l).expect("layer under n_layers");
            let record = store.record_bytes(l).expect("layer under n_layers");
            if slots == 0 || slots > layout.n_expert() || record > pool.buf_len() {
                return Err(GpuError::Driver(format!(
                    "expert disk backing: layer {l}: {slots} slots over {} experts, \
                     {record} byte records in {} byte staging buffers",
                    layout.n_expert(),
                    pool.buf_len()
                )));
            }
            lru.push(SlotLru::new(layout.n_expert(), slots));
        }
        Ok(Self {
            store,
            pool,
            lru,
            in_flight: Vec::new(),
            jobs: Vec::new(),
            broken: false,
            stats: MoeDiskStats::default(),
        })
    }

    pub(super) fn store(&self) -> &DiskExpertStore {
        &self.store
    }

    pub(super) fn stats(&self) -> MoeDiskStats {
        self.stats
    }

    /// Resolve one chunk's `rows` of artifact layer `layer`, fill its misses
    /// through `dev`, and write each row's slot into `slot_rows`. On `Ok` the
    /// copies are enqueued in stream order ahead of whatever the caller
    /// launches next on that stream; on `Err` no slot this call assigned is
    /// left looking valid.
    pub(super) fn fill(
        &mut self,
        layer: usize,
        rows: &[u32],
        slot_rows: &mut [u32],
        dev: &mut impl SlotCopier,
    ) -> Result<(), GpuError> {
        if self.broken {
            return Err(broken());
        }
        if layer >= self.lru.len() || slot_rows.len() != rows.len() {
            return Err(GpuError::Driver(format!(
                "expert disk backing: layer {layer} of {}, {} rows into {} slots",
                self.lru.len(),
                rows.len(),
                slot_rows.len()
            )));
        }
        // the previous fill's copies are done by now in practice (the caller
        // synced to read this launch's routing); waiting makes it certain
        self.drain(dev)?;
        let mut jobs = std::mem::take(&mut self.jobs);
        jobs.clear();
        self.lru[layer].resolve(rows, slot_rows, &mut jobs);
        self.stats.rows += rows.len() as u64;
        let r = self.fill_jobs(layer, &jobs, dev);
        if r.is_err() {
            self.lru[layer].invalidate(&jobs);
        }
        self.jobs = jobs;
        r
    }

    fn fill_jobs(
        &mut self,
        layer: usize,
        jobs: &[MissJob],
        dev: &mut impl SlotCopier,
    ) -> Result<(), GpuError> {
        let len = self
            .store
            .record_bytes(layer)
            .expect("layer checked by fill");
        let streams = self
            .store
            .layout(layer)
            .expect("layer checked by fill")
            .stream_bytes();
        for j in jobs {
            let lease = match self.pool.acquire() {
                Some(l) => l,
                None => {
                    self.stats.staging_waits += 1;
                    self.drain(dev)?;
                    self.pool.acquire().expect("drain released every lease")
                }
            };
            let t0 = Instant::now();
            let read = self.store.read_expert(
                layer,
                j.expert as usize,
                &mut self.pool.buf_mut(&lease)[..len],
            );
            self.stats.read_time += t0.elapsed();
            if let Err(e) = read {
                // nothing was enqueued from this buffer
                self.pool.release(lease);
                return Err(e.into());
            }
            self.stats.misses += 1;
            self.stats.bytes_read += len as u64;
            let buf = self.pool.buf(&lease);
            let mut off = 0;
            for (k, &n) in streams.iter().enumerate() {
                // SAFETY: the lease joins `in_flight` below, or on error is
                // held until a wait; a buffer is released only by `drain`,
                // after `wait` returned Ok.
                let r = unsafe { dev.enqueue(k, j.slot, &buf[off..off + n]) };
                if let Err(e) = r {
                    self.in_flight.push(lease);
                    return Err(e);
                }
                off += n;
            }
            self.in_flight.push(lease);
        }
        Ok(())
    }

    /// Wait for every enqueued copy and release its staging buffer.
    fn drain(&mut self, dev: &mut impl SlotCopier) -> Result<(), GpuError> {
        if self.in_flight.is_empty() {
            return Ok(());
        }
        let t0 = Instant::now();
        let r = dev.wait();
        self.stats.wait_time += t0.elapsed();
        if let Err(e) = r {
            // the device may still read these buffers: never reuse them
            self.broken = true;
            return Err(e);
        }
        for l in self.in_flight.drain(..) {
            self.pool.release(l);
        }
        Ok(())
    }
}

fn broken() -> GpuError {
    GpuError::Driver(
        "expert disk backing: an earlier slot copy failed to complete; reload the model".into(),
    )
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use paddock_models::ggml_type::GgmlType;

    use super::super::expert_store::{ExpertArtifactWriter, Identity};
    use super::super::moe_cache::{ExpertLayerLayout, ExpertPlaneLayout};
    use super::*;

    const ID: Identity = [3u8; 32];
    const N_EXPERT: usize = 8;

    fn layouts() -> Vec<ExpertLayerLayout> {
        let p = |rows, data, scales| {
            ExpertPlaneLayout::new(
                &[256, rows, N_EXPERT],
                GgmlType::Q4K,
                data * N_EXPERT,
                scales * N_EXPERT,
            )
            .expect("valid plane")
        };
        vec![
            ExpertLayerLayout::new([p(8, 64, 12), p(8, 64, 12), p(4, 32, 6)]).expect("layer 0"),
            ExpertLayerLayout::new([p(8, 96, 16), p(8, 96, 16), p(4, 48, 8)]).expect("layer 1"),
        ]
    }

    fn record(ls: &[ExpertLayerLayout], layer: usize, expert: usize) -> Vec<u8> {
        let len: usize = ls[layer].stream_bytes().iter().sum();
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31) ^ (layer as u8 * 101) ^ (expert as u8 * 17))
            .collect()
    }

    fn artifact(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pdk-md-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("temp dir");
        let path = d.join("experts.pdxa");
        let ls = layouts();
        let mut w = ExpertArtifactWriter::create(&path, &ID, &ls).expect("create");
        for l in 0..ls.len() {
            for e in 0..N_EXPERT {
                w.push_expert(&record(&ls, l, e)).expect("push");
            }
        }
        w.finish().expect("finish");
        path
    }

    fn experts(path: &Path, slots: usize, staging: usize) -> DiskExperts<Vec<u8>> {
        let store = DiskExpertStore::open(path, &ID, &layouts()).expect("open");
        let len = (0..store.n_layers())
            .filter_map(|l| store.record_bytes(l))
            .max()
            .expect("layers");
        let pool = StagingPool::new(staging, len, |n| Ok(vec![0u8; n])).expect("pool");
        DiskExperts::new(store, pool, slots).expect("experts")
    }

    /// A device that applies each copy only at `wait`, reading the source
    /// then - as a DMA would - so a staging buffer reused before the wait
    /// lands the wrong bytes in a slot.
    #[derive(Default)]
    struct FakeDevice {
        /// `[stream][slot]` contents.
        slots: Vec<Vec<Vec<u8>>>,
        pending: Vec<(usize, u32, *const u8, usize)>,
        copies: usize,
        waits: usize,
        fail_enqueue_at: Option<usize>,
        fail_wait: bool,
    }

    impl FakeDevice {
        fn new(n_slots: usize) -> Self {
            Self {
                slots: vec![vec![Vec::new(); n_slots]; 6],
                ..Self::default()
            }
        }

        /// The six streams of `slot`, concatenated.
        fn slot(&self, slot: u32) -> Vec<u8> {
            (0..6)
                .flat_map(|k| self.slots[k][slot as usize].clone())
                .collect()
        }
    }

    impl SlotCopier for FakeDevice {
        unsafe fn enqueue(&mut self, stream: usize, slot: u32, src: &[u8]) -> Result<(), GpuError> {
            if self.fail_enqueue_at == Some(self.copies) {
                return Err(GpuError::Driver("injected enqueue failure".into()));
            }
            self.copies += 1;
            self.pending.push((stream, slot, src.as_ptr(), src.len()));
            Ok(())
        }

        fn wait(&mut self) -> Result<(), GpuError> {
            self.waits += 1;
            if self.fail_wait {
                return Err(GpuError::Driver("injected wait failure".into()));
            }
            for (k, slot, p, n) in self.pending.drain(..) {
                // SAFETY: the SlotCopier contract keeps `p..p+n` alive and
                // unmodified until this wait returns.
                let src = unsafe { std::slice::from_raw_parts(p, n) };
                self.slots[k][slot as usize] = src.to_vec();
            }
            Ok(())
        }
    }

    /// Fill `rows` and check every row's slot holds its expert's record once
    /// the copies land.
    fn fill_and_check(
        x: &mut DiskExperts<Vec<u8>>,
        dev: &mut FakeDevice,
        layer: usize,
        rows: &[u32],
    ) -> Vec<u32> {
        let mut slot_rows = vec![MOE_CACHE_NONE; rows.len()];
        x.fill(layer, rows, &mut slot_rows, dev).expect("fill");
        dev.wait().expect("wait");
        let ls = layouts();
        for (&e, &s) in rows.iter().zip(&slot_rows) {
            assert_eq!(
                dev.slot(s),
                record(&ls, layer, e as usize),
                "expert {e} slot {s}"
            );
        }
        slot_rows
    }

    #[test]
    fn plan_chunks_keeps_each_range_within_the_slots() {
        let mut out = Vec::new();
        // decode: one token, one range
        plan_chunks(&[1, 2], 2, N_EXPERT, 4, &mut out).expect("plan");
        assert_eq!(out, vec![0..1]);
        // tokens 0-1 touch {0,1,2,3}; token 2 adds 4 -> new range
        plan_chunks(&[0, 1, 2, 3, 4, 0, 4, 5], 2, N_EXPERT, 4, &mut out).expect("plan");
        assert_eq!(out, vec![0..2, 2..4]);
        // repeated experts do not count twice
        plan_chunks(&[1, 1, 1, 2, 2, 1], 2, N_EXPERT, 2, &mut out).expect("plan");
        assert_eq!(out, vec![0..3]);
        assert!(
            plan_chunks(&[0, 8], 2, N_EXPERT, 4, &mut out).is_err(),
            "id range"
        );
        assert!(
            plan_chunks(&[0, 1], 2, N_EXPERT, 1, &mut out).is_err(),
            "slots < top-k"
        );
        assert!(
            plan_chunks(&[0, 1, 2], 2, N_EXPERT, 4, &mut out).is_err(),
            "ragged"
        );
    }

    #[test]
    fn lru_follows_the_resolve_kernel() {
        let mut lru = SlotLru::new(N_EXPERT, 3);
        let mut slots = [0u32; 4];
        let mut jobs = Vec::new();
        // empty slots are taken in index order; a duplicate row is one job
        lru.resolve(&[5, 6, 5, 7], &mut slots, &mut jobs);
        assert_eq!(slots, [0, 1, 0, 2]);
        assert_eq!(jobs.len(), 3);
        // 6 is the least recently used once 5 and 7 are touched again
        jobs.clear();
        lru.resolve(&[5, 7], &mut slots[..2], &mut jobs);
        assert!(jobs.is_empty());
        lru.resolve(&[1], &mut slots[..1], &mut jobs);
        assert_eq!(jobs, vec![MissJob { slot: 1, expert: 1 }]);
        assert_eq!(lru.slot_of[6], MOE_CACHE_NONE);
        // invalidation empties exactly what the jobs assigned
        lru.invalidate(&jobs);
        assert_eq!(lru.slot_of[1], MOE_CACHE_NONE);
        assert_eq!(lru.expert_in[1], MOE_CACHE_NONE);
        assert_eq!(lru.last_use[1], 0);
        assert_eq!(lru.slot_of[5], 0);
    }

    #[test]
    fn fills_misses_and_serves_hits_from_the_slots() {
        let path = artifact("fill");
        let mut x = experts(&path, 4, 2);
        let mut dev = FakeDevice::new(4);
        fill_and_check(&mut x, &mut dev, 0, &[1, 2, 1, 3]);
        let copies = dev.copies;
        assert_eq!(copies, 3 * 6, "three distinct experts, six streams each");
        // all hits: nothing read, nothing copied
        fill_and_check(&mut x, &mut dev, 0, &[3, 2]);
        assert_eq!(dev.copies, copies);
        // layer 1 has its own slots and record size
        let mut dev1 = FakeDevice::new(4);
        fill_and_check(&mut x, &mut dev1, 1, &[1, 2]);
        let s = x.stats();
        assert_eq!((s.rows, s.misses), (8, 5));
        let len0: usize = layouts()[0].stream_bytes().iter().sum();
        let len1: usize = layouts()[1].stream_bytes().iter().sum();
        assert_eq!(s.bytes_read, (3 * len0 + 2 * len1) as u64);
    }

    #[test]
    fn staging_is_never_reused_before_its_copy_completes() {
        // five misses through two buffers: two waits mid-fill, and the fake
        // applies copies at wait time, so early reuse would corrupt a slot
        let path = artifact("staging");
        let mut x = experts(&path, 6, 2);
        let mut dev = FakeDevice::new(6);
        fill_and_check(&mut x, &mut dev, 0, &[0, 1, 2, 3, 4]);
        assert_eq!(x.stats().staging_waits, 2);
        // the fifth miss's lease waits for the next fill's drain
        assert_eq!(x.pool.available(), 1);
        fill_and_check(&mut x, &mut dev, 0, &[4]);
        assert_eq!(x.pool.available(), 2);
    }

    #[test]
    fn a_failed_read_leaves_no_slot_looking_filled() {
        let path = artifact("read");
        let mut x = experts(&path, 4, 4);
        let mut dev = FakeDevice::new(4);
        fill_and_check(&mut x, &mut dev, 0, &[6]);
        // corrupt expert 2 of layer 0 on disk, after open
        let mut bytes = std::fs::read(&path).expect("read back");
        let first = x.store().record_offset(0, 2).expect("offset") as usize;
        bytes[first + 3] ^= 0x40;
        std::fs::write(&path, &bytes).expect("rewrite");
        let mut x = experts(&path, 4, 4);
        let mut dev = FakeDevice::new(4);
        let mut slot_rows = [MOE_CACHE_NONE; 3];
        // 1 reads fine and is enqueued, 2 fails: the whole chunk fails
        assert!(x.fill(0, &[1, 2, 6], &mut slot_rows, &mut dev).is_err());
        assert!(x.lru[0].slot_of.iter().all(|&s| s == MOE_CACHE_NONE));
        assert!(x.lru[0].expert_in.iter().all(|&e| e == MOE_CACHE_NONE));
        // the backing still serves: expert 1 is read again, not taken as a hit
        let misses = x.stats().misses;
        fill_and_check(&mut x, &mut dev, 0, &[1, 6]);
        assert_eq!(x.stats().misses, misses + 2);
        assert_eq!(x.pool.available() + x.in_flight.len(), 4, "no lease lost");
    }

    #[test]
    fn a_failed_enqueue_invalidates_and_keeps_the_buffer_until_a_wait() {
        let path = artifact("enqueue");
        let mut x = experts(&path, 4, 2);
        let mut dev = FakeDevice::new(4);
        dev.fail_enqueue_at = Some(8); // second expert, third stream
        let mut slot_rows = [MOE_CACHE_NONE; 2];
        assert!(x.fill(0, &[3, 4], &mut slot_rows, &mut dev).is_err());
        assert!(x.lru[0].slot_of.iter().all(|&s| s == MOE_CACHE_NONE));
        assert_eq!(x.pool.available(), 0, "both buffers wait for the device");
        dev.fail_enqueue_at = None;
        fill_and_check(&mut x, &mut dev, 0, &[3, 4]);
    }

    #[test]
    fn a_failed_wait_breaks_the_backing_and_strands_its_buffers() {
        let path = artifact("wait");
        let mut x = experts(&path, 4, 2);
        let mut dev = FakeDevice::new(4);
        let mut slot_rows = [MOE_CACHE_NONE; 2];
        x.fill(0, &[1, 2], &mut slot_rows, &mut dev).expect("fill");
        dev.fail_wait = true;
        // the next fill must wait for the last one's copies first
        assert!(x.fill(0, &[3], &mut slot_rows[..1], &mut dev).is_err());
        dev.fail_wait = false;
        assert!(x.fill(0, &[1], &mut slot_rows[..1], &mut dev).is_err());
        assert_eq!(
            x.pool.available(),
            0,
            "buffers the device may read are never reused"
        );
    }

    #[test]
    fn rejects_bad_requests_and_geometry() {
        let path = artifact("bad");
        let mut x = experts(&path, 4, 2);
        let mut dev = FakeDevice::new(4);
        let mut slot_rows = [MOE_CACHE_NONE; 2];
        assert!(
            x.fill(2, &[1, 2], &mut slot_rows, &mut dev).is_err(),
            "layer"
        );
        assert!(
            x.fill(0, &[1], &mut slot_rows, &mut dev).is_err(),
            "row count"
        );
        let store = || DiskExpertStore::open(&path, &ID, &layouts()).expect("open");
        let pool = |len| StagingPool::new(1, len, |n| Ok(vec![0u8; n])).expect("pool");
        let big = (0..2)
            .filter_map(|l| store().record_bytes(l))
            .max()
            .expect("layers");
        assert!(DiskExperts::new(store(), pool(big), 0).is_err(), "no slots");
        assert!(
            DiskExperts::new(store(), pool(big), N_EXPERT + 1).is_err(),
            "slots"
        );
        assert!(
            DiskExperts::new(store(), pool(big - 1), 4).is_err(),
            "staging too small"
        );
    }
}
