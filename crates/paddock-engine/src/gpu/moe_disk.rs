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
//! A launch whose distinct experts fit the slots (decode) runs the
//! token-batched pair once, the pinned cache's class. A wider one (prefill)
//! runs expert-major in waves of at most `slots` experts, like
//! `kq_moe_routed_waves` in qwen4exp: each touched expert is read at most
//! once per layer and launch, and the pair runs over the wave's pairs only.
//!
//! A slot whose fill did not complete is never consumed. A failed read or
//! enqueue invalidates every slot the launch assigned before the error
//! returns. A failed wait breaks the backing for good: the device may still
//! be reading the staging buffers, so they are never handed out again.
//!
//! The device is behind [`SlotCopier`] (enqueue a host-to-slot copy, mark the
//! copies so far, wait for a mark); the tests drive it with a recording
//! fake. A fill waits only for the copies whose staging buffers it needs,
//! and the routing goes up from pinned memory, so a wave's disk reads
//! overlap the kernels of the waves before it.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use cudarc::driver::{CudaSlice, DevicePtr};
use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MappedGguf;

use super::expert_store::{DiskExpertStore, Identity, StagingLease, StagingPool};
use super::host_plane::PinnedBuf;
use super::moe_cache::{
    CacheCopier, ExpertCache, ExpertLayerLayout, ExpertPlaneLayout, MOE_CACHE_NONE,
};
use super::{GpuError, GpuExecutor};

/// The default staging pool's ceiling, in records (about 128 MB pinned on
/// the 35B-A3B UD file).
const MAX_STAGING_RECORDS: usize = 64;

/// Threads a fill reads its misses on. PADDOCK_MOE_DISK_READERS overrides
/// it (development instrument).
const READERS: usize = 4;

fn dev_count(v: Result<String, std::env::VarError>, default: usize) -> usize {
    v.ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// The device half of a fill: copies from host staging memory into one
/// layer's slot planes, on the stream the expert kernels run on.
pub(super) trait SlotCopier {
    /// A point in the copy order (a CUDA event on the device).
    type Mark;

    /// Enqueue a copy of `src` into stream `stream` (fill-descriptor order:
    /// gate data, gate scales, up data, up scales, down data, down scales)
    /// of `slot`.
    ///
    /// # Safety
    /// `src` must stay allocated and unmodified until a [`wait`] on a
    /// [`mark`] taken after this call returns `Ok`; the copy may still be
    /// reading it until then.
    ///
    /// [`wait`]: SlotCopier::wait
    /// [`mark`]: SlotCopier::mark
    unsafe fn enqueue(&mut self, stream: usize, slot: u32, src: &[u8]) -> Result<(), GpuError>;

    /// Mark every copy enqueued so far.
    fn mark(&mut self) -> Result<Self::Mark, GpuError>;

    /// Block until every copy enqueued before `mark` has completed.
    fn wait(&mut self, mark: &Self::Mark) -> Result<(), GpuError>;
}

/// Counters for the disk backing since load, summed over layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoeDiskStats {
    /// Rows resolved (routed `(token, expert)` pairs), as the pinned
    /// cache's `stats()` counts them.
    pub rows: u64,
    /// Experts read from disk.
    pub misses: u64,
    /// Artifact bytes those reads returned (whole records).
    pub bytes_read: u64,
    /// Wall time in positioned reads and digest checks.
    pub read_time: Duration,
    /// Times a fill found every staging buffer in flight and waited.
    pub staging_waits: u64,
    /// Wall time the host spent waiting for slot copies to complete.
    pub wait_time: Duration,
    /// Misses served from the host tier (no disk read).
    pub tier_hits: u64,
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
    /// experts fit the slots (`plan_waves`), so a tick never evicts an
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
                .expect("plan_waves keeps a tick's distinct experts within the slots");
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

/// Assign a launch's distinct routed experts (`n_active` rows per token) to
/// waves of at most `slots` each, into `wave_of` (one entry per expert,
/// `MOE_CACHE_NONE` for an expert the launch does not route to), in id
/// order. Returns the wave count: 1 when every distinct expert fits the
/// slots at once. Waves set the order a token's experts are summed in, so
/// they depend on the routing alone, never on what the slots hold: the same
/// prompt gives the same logits whatever ran before it.
pub(super) fn plan_waves(
    ids: &[u32],
    n_active: usize,
    n_expert: usize,
    slots: usize,
    wave_of: &mut Vec<u32>,
) -> Result<usize, GpuError> {
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
    let mut routed = vec![false; n_expert];
    for &e in ids {
        routed[e as usize] = true;
    }
    wave_of.clear();
    wave_of.resize(n_expert, MOE_CACHE_NONE);
    let mut n = 0;
    for (i, e) in (0..n_expert).filter(|&e| routed[e]).enumerate() {
        wave_of[e] = (i / slots) as u32;
        n = i / slots + 1;
    }
    Ok(n)
}

/// Where one miss's record comes from in a fill.
#[derive(Clone, Copy)]
enum Src {
    /// Already in this host tier buffer.
    Hit(usize),
    /// Read into this tier buffer, which it is admitted to.
    Tier(usize),
    /// Read into this staging lease (an index into the batch's leases).
    Staging(usize),
}

/// Pinned host copies of hot expert records, a tier between the slots and
/// the disk: a hit is copied straight to its slot, with no read and no
/// second digest check. Each layer owns an equal share of the buffers. An
/// expert is admitted when it has been routed to at least as often as the
/// least routed expert it would replace; a buffer is never written while a
/// copy out of it may be pending.
struct HostTier<B> {
    bufs: Vec<B>,
    /// Buffers a layer owns: layer `l` owns `l * quota..(l + 1) * quota`.
    quota: usize,
    /// The expert each buffer holds.
    held: Vec<u32>,
    /// Per layer, the buffer each expert is in.
    at: Vec<Vec<u32>>,
    /// Per layer, the routed rows of each expert since load.
    count: Vec<Vec<u32>>,
    pending: Vec<bool>,
}

impl<B: AsRef<[u8]> + AsMut<[u8]>> HostTier<B> {
    fn new(bufs: Vec<B>, n_expert: Vec<usize>) -> Self {
        let quota = bufs.len() / n_expert.len().max(1);
        Self {
            quota,
            held: vec![MOE_CACHE_NONE; bufs.len()],
            pending: vec![false; bufs.len()],
            at: n_expert.iter().map(|&n| vec![MOE_CACHE_NONE; n]).collect(),
            count: n_expert.iter().map(|&n| vec![0; n]).collect(),
            bufs,
        }
    }

    fn note(&mut self, layer: usize, rows: &[u32]) {
        if self.quota == 0 {
            return;
        }
        for &e in rows {
            let c = &mut self.count[layer][e as usize];
            *c = c.saturating_add(1);
        }
    }

    fn lookup(&self, layer: usize, expert: u32) -> Option<usize> {
        let b = *self.at.get(layer)?.get(expert as usize)?;
        (b != MOE_CACHE_NONE).then_some(b as usize)
    }

    /// A buffer of `layer` for `expert`: an empty one, else the one holding
    /// the least routed expert if `expert` is routed at least as often.
    /// Never a pending buffer.
    fn admit(&mut self, layer: usize, expert: u32) -> Option<usize> {
        let count = &self.count[layer];
        let held = &self.held;
        let b = (layer * self.quota..(layer + 1) * self.quota)
            .filter(|&b| !self.pending[b])
            .min_by_key(|&b| match held[b] {
                MOE_CACHE_NONE => (0, 0),
                h => (1, count[h as usize]),
            })?;
        let old = self.held[b];
        if old != MOE_CACHE_NONE {
            if count[expert as usize] < count[old as usize] {
                return None;
            }
            self.at[layer][old as usize] = MOE_CACHE_NONE;
        }
        self.held[b] = expert;
        self.at[layer][expert as usize] = b as u32;
        Some(b)
    }

    /// Forget what buffer `b` holds.
    fn forget(&mut self, b: usize) {
        let e = std::mem::replace(&mut self.held[b], MOE_CACHE_NONE);
        if e != MOE_CACHE_NONE {
            let layer = b / self.quota;
            self.at[layer][e as usize] = MOE_CACHE_NONE;
        }
    }

    fn buf(&self, b: usize) -> &[u8] {
        self.bufs[b].as_ref()
    }

    /// The buffers `bs`, in their order. Only those are touched: others may
    /// be under a copy.
    fn bufs_mut(&mut self, bs: &[usize]) -> Vec<&mut [u8]> {
        let mut out: Vec<Option<&mut [u8]>> = bs.iter().map(|_| None).collect();
        for (i, b) in self.bufs.iter_mut().enumerate() {
            if let Some(at) = bs.iter().position(|&x| x == i) {
                out[at] = Some(b.as_mut());
            }
        }
        out.into_iter()
            .map(|b| b.expect("every index names a buffer"))
            .collect()
    }
}

/// The disk backing of one model: the artifact, the staging pool every layer
/// shares (layers run one at a time), and each MoE layer's slot LRU. `M` is
/// the device's copy mark.
pub(super) struct DiskExperts<B, M> {
    store: DiskExpertStore,
    pool: StagingPool<B>,
    tier: HostTier<B>,
    lru: Vec<SlotLru>,
    /// Leases and tier buffers whose copies were enqueued and not yet
    /// waited for, oldest first, each batch behind the mark taken after its
    /// copies.
    in_flight: VecDeque<(M, Vec<StagingLease>, Vec<usize>)>,
    jobs: Vec<MissJob>,
    /// Threads a fill reads its misses on.
    readers: usize,
    broken: bool,
    stats: MoeDiskStats,
}

impl<B: AsRef<[u8]> + AsMut<[u8]>, M> DiskExperts<B, M> {
    /// `slots` per layer over every layer of `store`. The pool's buffers and
    /// the host tier's (`tier`, split evenly between the layers; none turns
    /// it off) must each hold one record of the largest layer.
    pub(super) fn new(
        store: DiskExpertStore,
        pool: StagingPool<B>,
        tier: Vec<B>,
        slots: usize,
        readers: usize,
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
        if let Some(b) = tier.iter().find(|b| b.as_ref().len() < pool.buf_len()) {
            return Err(GpuError::Driver(format!(
                "expert disk backing: a {} byte host tier buffer for {} byte records",
                b.as_ref().len(),
                pool.buf_len()
            )));
        }
        let tier = HostTier::new(tier, lru.iter().map(|l| l.slot_of.len()).collect());
        Ok(Self {
            store,
            pool,
            tier,
            lru,
            in_flight: VecDeque::new(),
            jobs: Vec::new(),
            readers,
            broken: false,
            stats: MoeDiskStats::default(),
        })
    }

    #[cfg(test)]
    fn store(&self) -> &DiskExpertStore {
        &self.store
    }

    pub(super) fn stats(&self) -> MoeDiskStats {
        self.stats
    }

    /// Resolve one launch's (or wave's) `rows` of artifact layer `layer`, fill its misses
    /// through `dev`, and write each row's slot into `slot_rows`. On `Ok` the
    /// copies are enqueued in stream order ahead of whatever the caller
    /// launches next on that stream; on `Err` no slot this call assigned is
    /// left looking valid.
    pub(super) fn fill(
        &mut self,
        layer: usize,
        rows: &[u32],
        slot_rows: &mut [u32],
        dev: &mut impl SlotCopier<Mark = M>,
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
        // resolve needs ids in range and the distinct ones within the slots
        let lru = &self.lru[layer];
        let mut seen = vec![false; lru.slot_of.len()];
        let mut distinct = 0;
        for &e in rows {
            let Some(s) = seen.get_mut(e as usize) else {
                return Err(GpuError::Driver(format!(
                    "expert disk backing: routed expert {e} over {}",
                    lru.slot_of.len()
                )));
            };
            distinct += usize::from(!*s);
            *s = true;
        }
        if distinct > lru.expert_in.len() {
            return Err(GpuError::Driver(format!(
                "expert disk backing: {distinct} distinct experts over {} slots",
                lru.expert_in.len()
            )));
        }
        let mut jobs = std::mem::take(&mut self.jobs);
        jobs.clear();
        self.lru[layer].resolve(rows, slot_rows, &mut jobs);
        self.stats.rows += rows.len() as u64;
        self.tier.note(layer, rows);
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
        dev: &mut impl SlotCopier<Mark = M>,
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
        let mut todo = jobs;
        let mut srcs = Vec::new();
        let mut leases = Vec::new();
        let mut tiered = Vec::new();
        while !todo.is_empty() {
            // Each miss is served from the host tier, read into a tier buffer
            // it is admitted to, or read into staging, as many as staging
            // allows; the reads run side by side. With no staging free, wait
            // for the oldest batch's copies only, so the reads overlap
            // whatever the stream still runs.
            srcs.clear();
            leases.clear();
            tiered.clear();
            for j in todo {
                let src = if let Some(b) = self.tier.lookup(layer, j.expert) {
                    Src::Hit(b)
                } else if let Some(b) = self.tier.admit(layer, j.expert) {
                    Src::Tier(b)
                } else if let Some(l) = self.pool.acquire() {
                    leases.push(l);
                    Src::Staging(leases.len() - 1)
                } else {
                    break;
                };
                if let Src::Hit(b) | Src::Tier(b) = src {
                    // never written again until this batch's mark is waited for
                    self.tier.pending[b] = true;
                    tiered.push(b);
                }
                srcs.push(src);
            }
            if srcs.is_empty() {
                if self.in_flight.is_empty() {
                    return Err(GpuError::Driver(
                        "expert disk backing: no staging buffer is free or in flight".into(),
                    ));
                }
                self.stats.staging_waits += 1;
                self.drain_oldest(dev)?;
                continue;
            }
            let (batch, rest) = todo.split_at(srcs.len());
            todo = rest;
            let reads: Vec<MissJob> = batch
                .iter()
                .zip(&srcs)
                .filter(|(_, s)| !matches!(s, Src::Hit(_)))
                .map(|(j, _)| *j)
                .collect();
            let t0 = Instant::now();
            let read = {
                let store = &self.store;
                let mut staged = self.pool.bufs_mut(&leases).into_iter();
                let admitted: Vec<usize> = srcs
                    .iter()
                    .filter_map(|s| match s {
                        Src::Tier(b) => Some(*b),
                        _ => None,
                    })
                    .collect();
                let mut admitted = self.tier.bufs_mut(&admitted).into_iter();
                let mut bufs: Vec<&mut [u8]> = srcs
                    .iter()
                    .filter_map(|s| match s {
                        Src::Hit(_) => None,
                        Src::Tier(_) => admitted.next(),
                        Src::Staging(_) => staged.next(),
                    })
                    .collect();
                read_records(store, layer, &reads, &mut bufs, len, self.readers)
            };
            self.stats.read_time += t0.elapsed();
            if let Err(e) = read {
                // nothing was enqueued from this batch; what the admitted
                // buffers hold is unknown
                for l in leases.drain(..) {
                    self.pool.release(l);
                }
                for s in &srcs {
                    if let Src::Tier(b) = s {
                        self.tier.forget(*b);
                    }
                }
                for &b in &tiered {
                    self.tier.pending[b] = false;
                }
                return Err(GpuError::ExpertIo(e));
            }
            self.stats.misses += reads.len() as u64;
            self.stats.bytes_read += (reads.len() * len) as u64;
            self.stats.tier_hits += (srcs.len() - reads.len()) as u64;
            let mut sent = vec![false; leases.len()];
            let mut failed = None;
            'batch: for (j, src) in batch.iter().zip(&srcs) {
                let buf = match *src {
                    Src::Hit(b) | Src::Tier(b) => self.tier.buf(b),
                    Src::Staging(i) => {
                        sent[i] = true;
                        self.pool.buf(&leases[i])
                    }
                };
                let mut off = 0;
                for (k, &n) in streams.iter().enumerate() {
                    // SAFETY: the buffer joins `in_flight` behind a mark
                    // taken below, or is stranded with the backing broken; a
                    // staging buffer is released, and a tier buffer written
                    // again, only after a wait on its mark returned Ok.
                    let r = unsafe { dev.enqueue(k, j.slot, &buf[off..off + n]) };
                    if let Err(e) = r {
                        failed = Some(e);
                        break 'batch;
                    }
                    off += n;
                }
            }
            // nothing was enqueued from the batch's later staging buffers
            let mut batch_leases = Vec::with_capacity(leases.len());
            for (l, sent) in leases.drain(..).zip(sent) {
                if sent {
                    batch_leases.push(l);
                } else {
                    self.pool.release(l);
                }
            }
            match dev.mark() {
                Ok(m) => self
                    .in_flight
                    .push_back((m, batch_leases, std::mem::take(&mut tiered))),
                Err(e) => {
                    // copies the device may still run, with no way to wait
                    // for them: strand their buffers
                    self.broken = true;
                    return Err(failed.unwrap_or(e));
                }
            }
            if let Some(e) = failed {
                return Err(e);
            }
        }
        Ok(())
    }

    /// Wait for the copies of earlier fills. A failed wait breaks the backing
    /// here rather than in whatever sync the caller would otherwise hit
    /// first.
    pub(super) fn settle(&mut self, dev: &mut impl SlotCopier<Mark = M>) -> Result<(), GpuError> {
        if self.broken {
            return Err(broken());
        }
        self.drain(dev)
    }

    /// Wait for every enqueued copy and release its staging buffer.
    fn drain(&mut self, dev: &mut impl SlotCopier<Mark = M>) -> Result<(), GpuError> {
        let Some((last, _, _)) = self.in_flight.back() else {
            return Ok(());
        };
        let t0 = Instant::now();
        let r = dev.wait(last);
        self.stats.wait_time += t0.elapsed();
        if let Err(e) = r {
            // the device may still read these buffers: never reuse them
            self.broken = true;
            return Err(e);
        }
        for (_, leases, tiered) in self.in_flight.drain(..) {
            for l in leases {
                self.pool.release(l);
            }
            for b in tiered {
                self.tier.pending[b] = false;
            }
        }
        Ok(())
    }

    /// Wait for the oldest batch's copies and release its staging buffers.
    fn drain_oldest(&mut self, dev: &mut impl SlotCopier<Mark = M>) -> Result<(), GpuError> {
        let Some((mark, _, _)) = self.in_flight.front() else {
            return Ok(());
        };
        let t0 = Instant::now();
        let r = dev.wait(mark);
        self.stats.wait_time += t0.elapsed();
        if let Err(e) = r {
            self.broken = true;
            return Err(e);
        }
        let (_, leases, tiered) = self.in_flight.pop_front().expect("front checked above");
        for l in leases {
            self.pool.release(l);
        }
        for b in tiered {
            self.tier.pending[b] = false;
        }
        Ok(())
    }
}

/// Read the records of `jobs` into `bufs` (one each, `len` bytes used),
/// spread over up to `readers` threads: positioned reads share the file, and
/// each thread verifies its own records' digests.
fn read_records(
    store: &DiskExpertStore,
    layer: usize,
    jobs: &[MissJob],
    bufs: &mut [&mut [u8]],
    len: usize,
    readers: usize,
) -> std::io::Result<()> {
    let read = |jobs: &[MissJob], bufs: &mut [&mut [u8]]| {
        jobs.iter()
            .zip(bufs.iter_mut())
            .try_for_each(|(j, b)| store.read_expert(layer, j.expert as usize, &mut b[..len]))
    };
    let per = jobs.len().div_ceil(readers.max(1));
    if per >= jobs.len() {
        return read(jobs, bufs);
    }
    std::thread::scope(|s| {
        let threads: Vec<_> = jobs
            .chunks(per)
            .zip(bufs.chunks_mut(per))
            .map(|(j, b)| s.spawn(move || read(j, b)))
            .collect();
        // a panicked reader is a failed read: the fill invalidates its slots
        threads.into_iter().try_for_each(|t| {
            t.join()
                .unwrap_or_else(|_| Err(std::io::Error::other("an expert reader thread panicked")))
        })
    })
}

/// Why a qwen35-style family refuses to serve with disk-backed experts. One
/// sentence each, naming the cause and the way out; returned at setup, never
/// discovered mid-request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MoeDiskRefusal {
    /// An expert seat is not a k-quant plane (the artifact holds only the
    /// k-quant repack).
    Quant(GgmlType),
    /// The budget buys fewer slots than one token routes to: there is no
    /// zero-copy fallback on disk.
    Slots { slots: usize, n_active: usize },
    /// A path that captures CUDA graphs or keeps several sequences in
    /// flight; the miss path syncs the host between kernels.
    Mode(&'static str),
    /// A forward ran before `enable_moe_cache` seated the slot caches.
    NoCache,
}

impl std::fmt::Display for MoeDiskRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Quant(ty) => write!(
                f,
                "disk-backed MoE experts need k-quant expert tensors; this file has {ty:?} \
                 experts - serve it with pinned backing"
            ),
            Self::Slots { slots, n_active } => write!(
                f,
                "disk-backed MoE experts need a VRAM slot cache of at least {n_active} experts \
                 a layer (top-{n_active} routing); the budget buys {slots} - lower max_ctx or \
                 raise [moe_offload] vram_gb"
            ),
            Self::Mode(what) => write!(
                f,
                "disk-backed MoE experts serve one sequence on the eager path; {what} is not \
                 supported with them yet - serve this model with pinned backing for it"
            ),
            Self::NoCache => write!(
                f,
                "disk-backed MoE experts need enable_moe_cache before the first forward"
            ),
        }
    }
}

impl From<MoeDiskRefusal> for GpuError {
    fn from(r: MoeDiskRefusal) -> Self {
        GpuError::Unsupported(r.to_string())
    }
}

/// Slots per layer for a disk backing: as many as `budget` bytes buy at
/// `price` bytes a slot (or the development pin), at least `n_active` and at
/// most `n_expert`.
pub(crate) fn disk_slots(
    budget: u64,
    price: u64,
    pin: Option<usize>,
    n_active: usize,
    n_expert: usize,
) -> Result<usize, MoeDiskRefusal> {
    let slots = pin
        .unwrap_or_else(|| budget.checked_div(price).unwrap_or(0) as usize)
        .min(n_expert);
    if slots < n_active.max(1) {
        return Err(MoeDiskRefusal::Slots { slots, n_active });
    }
    Ok(slots)
}

impl GpuExecutor {
    /// The disk backing's seat for one expert tensor: the geometry
    /// [`GpuExecutor::try_repack_kquant`] would produce, from the GGUF shape
    /// and type alone, refused wherever that repack would refuse. `None` when
    /// the tensor is not a k-quant type.
    pub(crate) fn kq_expert_layout(
        &self,
        map: &MappedGguf,
        name: &str,
    ) -> Result<Option<ExpertPlaneLayout>, GpuError> {
        self.rotated_basis_guard(map, name)?;
        let (info, _) = map.tensor_bytes(name)?;
        let ty = info.ggml_type;
        if super::kq_params(ty).is_none() {
            return Ok(None);
        }
        // the repack's own pack checks (try_repack_kquant, repack_kquant_raw)
        if (super::kq_is_iq(ty) && !self.has_kquant_iq())
            || (super::kq_is_ternary(ty) && !self.has_kquant_ternary())
            || (matches!(ty, GgmlType::Q5_1 | GgmlType::Q8_0) && !self.has_kquant_flat32())
            || (ty == GgmlType::Q5_0 && !self.has_kquant_q50())
        {
            return Err(GpuError::NoKernel {
                name: name.to_owned(),
                ty,
            });
        }
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        let (data, scales) = super::kquant::kq_repacked_bytes(&dims, ty, name)?;
        ExpertPlaneLayout::new(&dims, ty, data, scales).map(Some)
    }
}

/// What an artifact must have been produced from: the kernel pack (it owns
/// the repack format), and for every expert tensor its name, type, shape and
/// a sample of its bytes (the head and tail 64 KiB), so a different file of
/// the same architecture is refused without hashing all of it.
pub(crate) fn expert_identity(
    map: &MappedGguf,
    names: &[String],
    pack_version: [u32; 3],
) -> Result<Identity, GpuError> {
    const SAMPLE: usize = 64 << 10;
    let mut h = blake3::Hasher::new();
    h.update(b"paddock moe expert artifact v1");
    for v in pack_version {
        h.update(&v.to_le_bytes());
    }
    for name in names {
        let (info, bytes) = map.tensor_bytes(name)?;
        h.update(name.as_bytes());
        h.update(&info.ggml_type.raw().to_le_bytes());
        for d in &info.dims {
            h.update(&d.to_le_bytes());
        }
        h.update(&(bytes.len() as u64).to_le_bytes());
        h.update(&bytes[..bytes.len().min(SAMPLE)]);
        h.update(&bytes[bytes.len().saturating_sub(SAMPLE)..]);
    }
    Ok(*h.finalize().as_bytes())
}

/// Write the artifact for the expert tensors `names` (gate, up, down per MoE
/// layer, in layer order) straight from the GGUF. One layer's planes are
/// repacked in VRAM at a time and copied down one expert record at a time,
/// so host memory holds one record whatever the model's size, and nothing
/// has to fit page-locked RAM.
pub(crate) fn write_expert_artifact(
    exec: &GpuExecutor,
    map: &MappedGguf,
    names: &[String],
    path: &Path,
) -> Result<(), GpuError> {
    if names.is_empty() || !names.len().is_multiple_of(3) {
        return Err(GpuError::Unsupported(
            "expert artifact: want gate, up and down names per MoE layer".into(),
        ));
    }
    let seat = |name: &String| match exec.kq_expert_layout(map, name)? {
        Some(l) => Ok(l),
        None => Err(GpuError::from(MoeDiskRefusal::Quant(
            map.tensor_bytes(name)?.0.ggml_type,
        ))),
    };
    let layouts = names
        .chunks(3)
        .map(|n| ExpertLayerLayout::new([seat(&n[0])?, seat(&n[1])?, seat(&n[2])?]))
        .collect::<Result<Vec<_>, _>>()?;
    let id = expert_identity(map, names, exec.pack_version())?;
    let mut w = super::expert_store::ExpertArtifactWriter::create(path, &id, &layouts)
        .map_err(GpuError::ExpertIo)?;
    let mut record = Vec::new();
    for (layer, layout) in names.chunks(3).zip(&layouts) {
        let mut planes = Vec::with_capacity(3);
        for (name, want) in layer.iter().zip(&layout.planes) {
            let p = exec
                .try_repack_kquant(map, name)?
                .ok_or_else(|| GpuError::Unsupported(format!("{name}: not a k-quant plane")))?;
            if ExpertPlaneLayout::new(&p.dims, p.ty, p.data.len(), p.scales.len())? != *want {
                return Err(GpuError::Driver(format!(
                    "expert artifact: {name} repacked to a geometry its layout does not predict"
                )));
            }
            planes.push(p);
        }
        let per = layout.stream_bytes();
        record.resize(per.iter().sum(), 0);
        for e in 0..layout.n_expert() {
            let mut at = 0;
            for (i, p) in planes.iter().enumerate() {
                for (j, s) in [&p.data, &p.scales].into_iter().enumerate() {
                    let n = per[2 * i + j];
                    exec.stream
                        .memcpy_dtoh(&s.slice(e * n..(e + 1) * n), &mut record[at..at + n])
                        .map_err(super::from_driver)?;
                    at += n;
                }
            }
            exec.stream.synchronize().map_err(super::from_driver)?;
            w.push_expert(&record).map_err(GpuError::ExpertIo)?;
        }
    }
    w.finish().map_err(GpuError::ExpertIo)
}

/// One model's disk backing, shared by its MoE layers. Opened at load
/// against the model's layouts; the slot LRUs are seated with the caches.
pub(crate) struct MoeDiskBacking {
    layouts: Vec<ExpertLayerLayout>,
    /// `None` only if seating failed part-way; the model is then reloaded.
    state: Mutex<Option<Seat>>,
    scratch: Mutex<Scratch>,
    /// Set by the first `seat`.
    routing: OnceLock<DiskRouting>,
}

/// One launch's slot routing on the device, shared by every layer (they run
/// one at a time): each pair's slot, `MOE_CACHE_NONE` outside the wave, and
/// for a wave its pair and token lists with their counts, as the LIST pair
/// kernels take them.
pub(crate) struct DiskRouting {
    pub(crate) slots: CudaSlice<u32>,
    pub(crate) pairs: CudaSlice<u32>,
    pub(crate) n_pairs: CudaSlice<u32>,
    pub(crate) tokens: CudaSlice<u32>,
    pub(crate) n_tokens: CudaSlice<u32>,
    /// Pinned sources for the uploads, so they never sync the stream.
    host: Mutex<HostRouting>,
}

/// Two pinned upload buffers, used in turn: one is written while the
/// other's copies may still be pending. Each holds every routing word of
/// one upload.
struct HostRouting {
    bufs: [PinnedBuf; 2],
    /// Recorded after the copies out of each buffer.
    done: [Option<cudarc::driver::CudaEvent>; 2],
    next: usize,
    /// A wait on an earlier upload failed: a buffer may still be read by a
    /// copy with nothing left to wait on, so nothing is written again.
    broken: bool,
}

impl DiskRouting {
    fn new(exec: &GpuExecutor, max_rows: usize, max_tokens: usize) -> Result<Self, GpuError> {
        let words = 2 * max_rows + max_tokens + 2;
        let pinned = || {
            PinnedBuf::new(exec.stream.clone(), 4 * words)
                .map_err(|e| GpuError::Driver(format!("expert routing: {e}")))
        };
        Ok(Self {
            slots: exec.to_device_u32(&vec![MOE_CACHE_NONE; max_rows])?,
            pairs: exec.to_device_u32(&vec![0; max_rows])?,
            n_pairs: exec.to_device_u32(&[0])?,
            tokens: exec.to_device_u32(&vec![0; max_tokens])?,
            n_tokens: exec.to_device_u32(&[0])?,
            host: Mutex::new(HostRouting {
                bufs: [pinned()?, pinned()?],
                done: [None, None],
                next: 0,
                broken: false,
            }),
        })
    }

    /// Upload `slots`, and a wave's `pairs` and `tokens` with their counts,
    /// in stream order.
    fn upload(
        &self,
        exec: &GpuExecutor,
        slots: &[u32],
        wave: Option<(&[u32], &[u32])>,
    ) -> Result<(), GpuError> {
        let mut h = lock(&self.host);
        if h.broken {
            return Err(broken());
        }
        let i = h.next;
        // the copies out of this buffer, two uploads back
        if let Some(ev) = &h.done[i]
            && let Err(e) = ev.synchronize()
        {
            h.broken = true;
            return Err(super::error::from_driver(e));
        }
        h.done[i] = None;
        h.next ^= 1;
        let (pairs, tokens) = wave.unwrap_or((&[], &[]));
        let counts = [pairs.len() as u32, tokens.len() as u32];
        let mut parts: Vec<(&CudaSlice<u32>, &[u32])> = vec![(&self.slots, slots)];
        if wave.is_some() {
            parts.extend([
                (&self.pairs, pairs),
                (&self.tokens, tokens),
                (&self.n_pairs, &counts[..1]),
                (&self.n_tokens, &counts[1..]),
            ]);
        }
        let HostRouting {
            bufs, done, broken, ..
        } = &mut *h;
        let buf = bufs[i].as_mut();
        let mut at = 0;
        let mut r = Ok(());
        for (dst, src) in parts {
            let n = 4 * src.len();
            if src.len() > dst.len() || at + n > buf.len() {
                r = Err(GpuError::Driver(format!(
                    "expert disk backing: {} routing words over a {}-word buffer",
                    src.len(),
                    dst.len()
                )));
                break;
            }
            for (b, w) in buf[at..at + n].as_chunks_mut::<4>().0.iter_mut().zip(src) {
                b.copy_from_slice(&w.to_ne_bytes());
            }
            // a write through a read view: the engine runs with event
            // tracking off, so the access kind records nothing
            let (p, _g) = dst.device_ptr(&exec.stream);
            // SAFETY: `dst` holds at least `src.len()` u32s (checked above),
            // and the copy is ordered on the stream after every kernel
            // already reading it. Its pinned source is not written again
            // until the event recorded below has completed.
            r = unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    p,
                    &buf[at..at + n],
                    exec.stream.cu_stream(),
                )
            }
            .map_err(super::error::from_driver);
            if r.is_err() {
                break;
            }
            at += n;
        }
        match exec.record_event() {
            Ok(ev) => done[i] = Some(ev),
            // no event to wait on before the next write: wait now
            Err(e) => {
                if let Err(s) = exec.stream.synchronize() {
                    *broken = true;
                    return Err(super::error::from_driver(s));
                }
                r = r.and(Err(e));
            }
        }
        r
    }
}

impl MoeDiskBacking {
    /// Open the artifact at `path` and check it against `identity` and the
    /// model's per-layer `layouts`.
    pub(crate) fn open(
        path: &Path,
        identity: &Identity,
        layouts: Vec<ExpertLayerLayout>,
    ) -> Result<Self, GpuError> {
        let store = DiskExpertStore::open(path, identity, &layouts).map_err(|e| {
            GpuError::Unsupported(format!("disk-backed MoE experts: {}: {e}", path.display()))
        })?;
        Ok(Self {
            layouts,
            state: Mutex::new(Some(Seat::Open(store))),
            scratch: Mutex::new(Scratch::default()),
            routing: OnceLock::new(),
        })
    }

    /// Bytes one slot costs in every layer together.
    pub(crate) fn slot_price(&self) -> u64 {
        self.layouts
            .iter()
            .map(|l| l.stream_bytes().iter().sum::<usize>() as u64)
            .sum()
    }

    /// Seat `slots` per layer: the caches' LRUs, the pinned staging pool,
    /// and routing for launches of up to `max_tokens` tokens at top-`n_active`.
    /// Returns one slot cache per layer, in layer order.
    pub(crate) fn seat(
        &self,
        exec: &GpuExecutor,
        slots: usize,
        max_tokens: usize,
        n_active: usize,
    ) -> Result<Vec<ExpertCache>, GpuError> {
        // the prefill waves' pair kernels, checked here rather than mid-request
        if exec.kernels.kquant_moe_gate_up_list.is_none() {
            return Err(GpuError::MissingOp("kquant_moe_gate_up_list"));
        }
        if exec.kernels.kquant_moe_down_list.is_none() {
            return Err(GpuError::MissingOp("kquant_moe_down_list"));
        }
        let max_rows = max_tokens * n_active;
        match self.routing.get() {
            Some(r) if r.slots.len() < max_rows || r.tokens.len() < max_tokens => {
                return Err(GpuError::Driver(format!(
                    "expert disk backing: seated for {} routed rows, asked for {max_rows}",
                    r.slots.len()
                )));
            }
            Some(_) => {}
            None => {
                let _ = self
                    .routing
                    .set(DiskRouting::new(exec, max_rows, max_tokens)?);
            }
        }
        let mut caches = Vec::with_capacity(self.layouts.len());
        for l in &self.layouts {
            caches.push(exec.new_expert_cache_for_disk(l.clone(), slots)?);
        }
        let mut state = lock(&self.state);
        // a re-seat starts from empty slots
        let store = match state.take() {
            Some(Seat::Open(s)) => s,
            Some(Seat::Seated(x)) => x.store,
            None => return Err(broken()),
        };
        let len = (0..store.n_layers())
            .filter_map(|l| store.record_bytes(l))
            .max()
            .expect("an artifact has at least one layer");
        let stream = exec.stream.clone();
        // Staging buffers, one expert record each (about 2 MB on the 35B-A3B
        // UD file), bound the host memory the miss path pins. One wave's
        // misses at most, so a wave's reads overlap the previous wave's
        // kernels (cold TTFT at 4000 tokens and 32 slots: 8 buffers 26.6 s,
        // 32 20.5 s, 64 20.2 s). PADDOCK_MOE_DISK_STAGING overrides it
        // (development instrument).
        let records = dev_count(
            paddock_models::dev_var!("PADDOCK_MOE_DISK_STAGING"),
            slots.min(MAX_STAGING_RECORDS),
        );
        let readers = dev_count(
            paddock_models::dev_var!("PADDOCK_MOE_DISK_READERS"),
            READERS,
        );
        let pool = StagingPool::new(records, len, |n| PinnedBuf::new(stream.clone(), n))
            .map_err(|e| GpuError::Driver(format!("expert staging: {e}")))?;
        // The host tier: PADDOCK_MOE_DISK_HOST_MB of pinned records
        // (development instrument; none by default).
        let tier_mb = dev_count(paddock_models::dev_var!("PADDOCK_MOE_DISK_HOST_MB"), 0);
        let tier = (0..(tier_mb << 20) / len)
            .map(|_| PinnedBuf::new(stream.clone(), len))
            .collect::<std::io::Result<Vec<_>>>()
            .map_err(|e| GpuError::Driver(format!("expert host tier: {e}")))?;
        *state = Some(Seat::Seated(Box::new(DiskExperts::new(
            store, pool, tier, slots, readers,
        )?)));
        Ok(caches)
    }

    /// Counters since load; zero before `seat`.
    pub(crate) fn stats(&self) -> MoeDiskStats {
        match &*lock(&self.state) {
            Some(Seat::Seated(x)) => x.stats(),
            _ => MoeDiskStats::default(),
        }
    }

    /// Plan one launch of `layer`: download its `rows` routed ids from `idx`
    /// (a host sync) and assign its experts to waves that fit the cache.
    /// Returns the wave count; each wave is then served by [`fill_wave`] and
    /// the caller's expert kernels over the routing it returns.
    ///
    /// [`fill_wave`]: MoeDiskBacking::fill_wave
    pub(crate) fn plan(
        &self,
        exec: &GpuExecutor,
        cache: &ExpertCache,
        idx: &CudaSlice<u32>,
        rows: usize,
        n_active: usize,
    ) -> Result<usize, GpuError> {
        if let Some(Seat::Seated(x)) = &mut *lock(&self.state) {
            x.settle(&mut CacheCopier { exec, cache })?;
        }
        let mut sc = lock(&self.scratch);
        sc.n_waves = 0;
        sc.ids = exec.to_host_u32_len(idx, rows)?;
        let Scratch { ids, wave_of, .. } = &mut *sc;
        let n = plan_waves(ids, n_active, cache.n_expert, cache.slots, wave_of)?;
        sc.n_waves = n;
        Ok(n)
    }

    /// Fill wave `wave` (from the last `plan`) of `layer` into `cache` and
    /// upload its routing: every row's slot when the launch has one wave,
    /// else the wave's (its pairs to their slots, the rest absent) with its
    /// pair and token lists.
    pub(crate) fn fill_wave(
        &self,
        exec: &GpuExecutor,
        cache: &ExpertCache,
        layer: usize,
        n_active: usize,
        wave: usize,
    ) -> Result<&DiskRouting, GpuError> {
        let dr = self.routing.get().ok_or(MoeDiskRefusal::NoCache)?;
        let mut state = lock(&self.state);
        let Some(Seat::Seated(x)) = &mut *state else {
            return Err(MoeDiskRefusal::NoCache.into());
        };
        let mut sc = lock(&self.scratch);
        if wave >= sc.n_waves {
            return Err(GpuError::Driver(format!(
                "expert disk backing: wave {wave} of {} planned",
                sc.n_waves
            )));
        }
        let mut dev = CacheCopier { exec, cache };
        let Scratch {
            ids,
            wave_of,
            n_waves,
            rows,
            slots,
            routing,
            pairs,
            tokens,
        } = &mut *sc;
        if *n_waves == 1 {
            slots.clear();
            slots.resize(ids.len(), MOE_CACHE_NONE);
            x.fill(layer, ids, slots, &mut dev)?;
            dr.upload(exec, slots, None)?;
            return Ok(dr);
        }
        let w = wave as u32;
        pairs.clear();
        rows.clear();
        tokens.clear();
        for (p, &e) in ids.iter().enumerate() {
            if wave_of[e as usize] == w {
                pairs.push(p as u32);
                rows.push(e);
                let t = (p / n_active) as u32;
                if tokens.last() != Some(&t) {
                    tokens.push(t);
                }
            }
        }
        slots.clear();
        slots.resize(rows.len(), MOE_CACHE_NONE);
        x.fill(layer, rows, slots, &mut dev)?;
        routing.clear();
        routing.resize(ids.len(), MOE_CACHE_NONE);
        for (&p, &s) in pairs.iter().zip(slots.iter()) {
            routing[p as usize] = s;
        }
        dr.upload(exec, routing, Some((pairs, tokens)))?;
        Ok(dr)
    }
}

/// Per-launch host scratch of a disk backing.
#[derive(Default)]
struct Scratch {
    /// The launch's routed ids, one per `(token, expert)` pair.
    ids: Vec<u32>,
    /// Each expert's wave, and the wave count, from `plan_waves`.
    wave_of: Vec<u32>,
    n_waves: usize,
    /// One wave's routed ids and the slots they resolved to.
    rows: Vec<u32>,
    slots: Vec<u32>,
    /// The wave's routing over every pair, its pairs and its tokens.
    routing: Vec<u32>,
    pairs: Vec<u32>,
    tokens: Vec<u32>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Drop for MoeDiskBacking {
    /// The counters at unload, so every run reports them without a probe.
    fn drop(&mut self) {
        let s = self.stats();
        if s.rows > 0 {
            tracing::info!(
                rows = s.rows,
                misses = s.misses,
                hit_pct = 100.0 * (1.0 - s.misses as f64 / s.rows as f64),
                read_mib = s.bytes_read >> 20,
                read_mib_s = (s.bytes_read >> 20) as f64 / s.read_time.as_secs_f64().max(1e-9),
                read_s = s.read_time.as_secs_f64(),
                staging_waits = s.staging_waits,
                wait_s = s.wait_time.as_secs_f64(),
                "disk-backed MoE experts: totals at unload"
            );
        }
    }
}

enum Seat {
    Open(DiskExpertStore),
    Seated(Box<DiskExperts<PinnedBuf, cudarc::driver::CudaEvent>>),
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

    fn experts(path: &Path, slots: usize, staging: usize) -> DiskExperts<Vec<u8>, usize> {
        let store = DiskExpertStore::open(path, &ID, &layouts()).expect("open");
        let len = (0..store.n_layers())
            .filter_map(|l| store.record_bytes(l))
            .max()
            .expect("layers");
        let pool = StagingPool::new(staging, len, |n| Ok(vec![0u8; n])).expect("pool");
        DiskExperts::new(store, pool, Vec::new(), slots, 2).expect("experts")
    }

    /// A device that applies each copy only at a `wait` on a later mark,
    /// reading the source then - as a DMA would - so a staging buffer reused
    /// before its wait lands the wrong bytes in a slot. A mark is the count
    /// of copies enqueued before it.
    #[derive(Default)]
    struct FakeDevice {
        /// `[stream][slot]` contents.
        slots: Vec<Vec<Vec<u8>>>,
        /// `(copy number, stream, slot, source)` not yet applied.
        pending: Vec<(usize, usize, u32, *const u8, usize)>,
        copies: usize,
        waits: usize,
        fail_enqueue_at: Option<usize>,
        fail_wait: bool,
        fail_mark: bool,
    }

    impl FakeDevice {
        fn new(n_slots: usize) -> Self {
            Self {
                slots: vec![vec![Vec::new(); n_slots]; 6],
                ..Self::default()
            }
        }

        /// Apply every pending copy, as a stream sync would.
        fn flush(&mut self) {
            self.wait(&self.copies.clone()).expect("wait");
        }

        /// The six streams of `slot`, concatenated.
        fn slot(&self, slot: u32) -> Vec<u8> {
            (0..6)
                .flat_map(|k| self.slots[k][slot as usize].clone())
                .collect()
        }
    }

    impl SlotCopier for FakeDevice {
        type Mark = usize;

        unsafe fn enqueue(&mut self, stream: usize, slot: u32, src: &[u8]) -> Result<(), GpuError> {
            if self.fail_enqueue_at == Some(self.copies) {
                return Err(GpuError::Driver("injected enqueue failure".into()));
            }
            self.pending
                .push((self.copies, stream, slot, src.as_ptr(), src.len()));
            self.copies += 1;
            Ok(())
        }

        fn mark(&mut self) -> Result<usize, GpuError> {
            if self.fail_mark {
                return Err(GpuError::Driver("injected mark failure".into()));
            }
            Ok(self.copies)
        }

        fn wait(&mut self, mark: &usize) -> Result<(), GpuError> {
            self.waits += 1;
            if self.fail_wait {
                return Err(GpuError::Driver("injected wait failure".into()));
            }
            let (done, left) = std::mem::take(&mut self.pending)
                .into_iter()
                .partition(|c| c.0 < *mark);
            self.pending = left;
            let done: Vec<_> = done;
            for (_, k, slot, p, n) in done {
                // SAFETY: the SlotCopier contract keeps `p..p+n` alive and
                // unmodified until a wait on a later mark returns.
                let src = unsafe { std::slice::from_raw_parts(p, n) };
                self.slots[k][slot as usize] = src.to_vec();
            }
            Ok(())
        }
    }

    /// `experts` with a host tier of `tier` buffers (split over both layers).
    fn tiered(
        path: &Path,
        slots: usize,
        staging: usize,
        tier: usize,
    ) -> DiskExperts<Vec<u8>, usize> {
        let store = DiskExpertStore::open(path, &ID, &layouts()).expect("open");
        let len = (0..store.n_layers())
            .filter_map(|l| store.record_bytes(l))
            .max()
            .expect("layers");
        let pool = StagingPool::new(staging, len, |n| Ok(vec![0u8; n])).expect("pool");
        DiskExperts::new(store, pool, vec![vec![0u8; len]; tier], slots, 2).expect("experts")
    }

    /// Fill `rows` and check every row's slot holds its expert's record once
    /// the copies land.
    fn fill_and_check(
        x: &mut DiskExperts<Vec<u8>, usize>,
        dev: &mut FakeDevice,
        layer: usize,
        rows: &[u32],
    ) -> Vec<u32> {
        let mut slot_rows = vec![MOE_CACHE_NONE; rows.len()];
        x.fill(layer, rows, &mut slot_rows, dev).expect("fill");
        dev.flush();
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
    fn plan_waves_fits_each_wave_in_the_slots() {
        let mut w = Vec::new();
        // decode: one token, one wave
        assert_eq!(
            plan_waves(&[1, 2], 2, N_EXPERT, 4, &mut w).expect("plan"),
            1
        );
        assert_eq!(
            &w[..3],
            &[MOE_CACHE_NONE, 0, 0],
            "an expert not routed to has no wave"
        );
        // six distinct experts over four slots: two waves, in id order
        let ids = [5, 1, 2, 3, 4, 0, 4, 5];
        assert_eq!(plan_waves(&ids, 2, N_EXPERT, 4, &mut w).expect("plan"), 2);
        assert_eq!(&w[..6], &[0, 0, 0, 0, 1, 1]);
        // repeated experts do not count twice
        assert_eq!(
            plan_waves(&[1, 1, 1, 2, 2, 1], 2, N_EXPERT, 2, &mut w).expect("plan"),
            1
        );
        assert!(
            plan_waves(&[0, 8], 2, N_EXPERT, 4, &mut w).is_err(),
            "id range"
        );
        assert!(
            plan_waves(&[0, 1], 2, N_EXPERT, 1, &mut w).is_err(),
            "slots < top-k"
        );
        assert!(
            plan_waves(&[0, 1, 2], 2, N_EXPERT, 4, &mut w).is_err(),
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
        // the fifth miss's lease waits for a drain: a fill does not wait for
        // earlier copies it does not need buffers from
        assert_eq!(x.pool.available(), 1);
        fill_and_check(&mut x, &mut dev, 0, &[4]);
        assert_eq!(x.pool.available(), 1);
        x.settle(&mut dev).expect("settle");
        assert_eq!(x.pool.available(), 2);
    }

    #[test]
    fn a_tier_hit_is_copied_without_a_read() {
        let path = artifact("tier-hit");
        // two slots, four tier buffers (two a layer)
        let mut x = tiered(&path, 2, 2, 4);
        let mut dev = FakeDevice::new(2);
        fill_and_check(&mut x, &mut dev, 0, &[1, 2]);
        assert_eq!((x.stats().misses, x.stats().tier_hits), (2, 0));
        // 3 and 4 evict 1 and 2 from the slots; the tier is full, so they
        // are admitted only when routed as often (they are: once each)
        fill_and_check(&mut x, &mut dev, 0, &[3, 4]);
        // 1 comes back from the tier only if it is still there
        let before = x.stats();
        fill_and_check(&mut x, &mut dev, 0, &[3, 1]);
        let after = x.stats();
        assert_eq!(
            (after.misses - before.misses) + (after.tier_hits - before.tier_hits),
            1,
            "3 is in a slot, 1 is a miss"
        );
        // routed most, 1 stays in the tier: a later miss of it reads nothing
        for _ in 0..3 {
            fill_and_check(&mut x, &mut dev, 0, &[1, 2]);
            fill_and_check(&mut x, &mut dev, 0, &[5, 6]);
        }
        let before = x.stats();
        fill_and_check(&mut x, &mut dev, 0, &[1, 2]);
        let after = x.stats();
        assert_eq!(after.misses, before.misses, "1 and 2 come from the tier");
        assert_eq!(after.tier_hits - before.tier_hits, 2);
    }

    #[test]
    fn a_pending_tier_buffer_is_never_written() {
        let path = artifact("tier-pending");
        // one tier buffer a layer, no settle between fills: the buffer
        // holding 1 is still under its copy when 2 misses
        let mut x = tiered(&path, 1 + 1, 2, 2);
        let mut dev = FakeDevice::new(2);
        let mut slots = [MOE_CACHE_NONE; 1];
        x.fill(0, &[1], &mut slots, &mut dev).expect("fill");
        assert_eq!(x.tier.lookup(0, 1), Some(0));
        x.fill(0, &[2], &mut slots, &mut dev).expect("fill");
        assert_eq!(x.tier.lookup(0, 1), Some(0), "the pending buffer kept 1");
        assert_eq!(x.tier.lookup(0, 2), None, "2 went through staging");
        dev.flush();
        let ls = layouts();
        assert_eq!(dev.slot(slots[0]), record(&ls, 0, 2));
        x.settle(&mut dev).expect("settle");
        assert!(x.tier.pending.iter().all(|&p| !p));
    }

    #[test]
    fn only_the_oldest_batch_is_waited_for_and_released() {
        // one miss, then two: the second fill finds one buffer free, sends
        // a batch of one, then must wait for the first fill's batch alone
        let path = artifact("oldest");
        let mut x = experts(&path, 4, 2);
        let mut dev = FakeDevice::new(4);
        let mut one = [MOE_CACHE_NONE; 1];
        x.fill(0, &[5], &mut one, &mut dev).expect("fill");
        let mut two = [MOE_CACHE_NONE; 2];
        x.fill(0, &[6, 7], &mut two, &mut dev).expect("fill");
        assert_eq!(x.stats().staging_waits, 1);
        // expert 5's copies landed; 6's are applied only once 7's buffer
        // came from 5's, so 6 and 7 are still pending behind their marks
        let ls = layouts();
        assert_eq!(dev.slot(one[0]), record(&ls, 0, 5));
        let held: usize = x.in_flight.iter().map(|(_, l, _)| l.len()).sum();
        assert_eq!(held, 2, "the two later batches stay in flight");
        dev.flush();
        assert_eq!(dev.slot(two[0]), record(&ls, 0, 6));
        assert_eq!(dev.slot(two[1]), record(&ls, 0, 7));
    }

    #[test]
    fn a_failed_mark_breaks_the_backing_and_strands_the_batch() {
        let path = artifact("mark");
        let mut x = experts(&path, 4, 2);
        let mut dev = FakeDevice::new(4);
        dev.fail_mark = true;
        let mut slot_rows = [MOE_CACHE_NONE; 2];
        assert!(x.fill(0, &[1, 2], &mut slot_rows, &mut dev).is_err());
        assert!(x.lru[0].slot_of.iter().all(|&s| s == MOE_CACHE_NONE));
        dev.fail_mark = false;
        assert!(x.fill(0, &[3], &mut slot_rows[..1], &mut dev).is_err());
        assert_eq!(
            x.pool.available(),
            0,
            "copies with no mark are never reused"
        );
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
        // 1 reads fine, 2 fails: the batch fails before anything is enqueued
        assert!(x.fill(0, &[1, 2, 6], &mut slot_rows, &mut dev).is_err());
        assert!(x.lru[0].slot_of.iter().all(|&s| s == MOE_CACHE_NONE));
        assert!(x.lru[0].expert_in.iter().all(|&e| e == MOE_CACHE_NONE));
        // the backing still serves: expert 1 is read again, not taken as a hit
        let misses = x.stats().misses;
        fill_and_check(&mut x, &mut dev, 0, &[1, 6]);
        assert_eq!(x.stats().misses, misses + 2);
        let held: usize = x.in_flight.iter().map(|(_, l, _)| l.len()).sum();
        assert_eq!(x.pool.available() + held, 4, "no lease lost");
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
        let mut five = [MOE_CACHE_NONE; 5];
        assert!(
            x.fill(0, &[0, 1, 2, 3, 4], &mut five, &mut dev).is_err(),
            "more distinct experts than slots"
        );
        assert!(
            x.fill(0, &[1, 8], &mut slot_rows, &mut dev).is_err(),
            "expert id"
        );
        let store = || DiskExpertStore::open(&path, &ID, &layouts()).expect("open");
        let pool = |len| StagingPool::new(1, len, |n| Ok(vec![0u8; n])).expect("pool");
        let big = (0..2)
            .filter_map(|l| store().record_bytes(l))
            .max()
            .expect("layers");
        assert!(
            DiskExperts::<_, usize>::new(store(), pool(big), Vec::new(), 0, 1).is_err(),
            "no slots"
        );
        assert!(
            DiskExperts::<_, usize>::new(store(), pool(big), Vec::new(), N_EXPERT + 1, 1).is_err(),
            "slots"
        );
        assert!(
            DiskExperts::<_, usize>::new(store(), pool(big - 1), Vec::new(), 4, 1).is_err(),
            "staging too small"
        );
    }

    /// A GGUF v3 file of Q4_K tensors `[256, 2, 4]` (8 superblocks, 1152
    /// bytes each), tensor `i` filled with byte `fill[i]`.
    fn gguf(dir: &Path, names: &[&str], fill: &[u8]) -> PathBuf {
        const BYTES: usize = 8 * 144;
        let mut h = Vec::new();
        h.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        h.extend_from_slice(&3u32.to_le_bytes());
        h.extend_from_slice(&(names.len() as u64).to_le_bytes());
        h.extend_from_slice(&0u64.to_le_bytes());
        for (i, n) in names.iter().enumerate() {
            h.extend_from_slice(&(n.len() as u64).to_le_bytes());
            h.extend_from_slice(n.as_bytes());
            h.extend_from_slice(&3u32.to_le_bytes());
            for d in [256u64, 2, 4] {
                h.extend_from_slice(&d.to_le_bytes());
            }
            h.extend_from_slice(&GgmlType::Q4K.raw().to_le_bytes());
            h.extend_from_slice(&((i * BYTES) as u64).to_le_bytes());
        }
        h.resize(h.len().next_multiple_of(32), 0);
        for &f in fill {
            h.extend(std::iter::repeat_n(f, BYTES));
        }
        let path = dir.join(format!(
            "m{}.gguf",
            fill.iter().map(|f| f.to_string()).collect::<String>()
        ));
        std::fs::write(&path, h).expect("write gguf");
        path
    }

    #[test]
    fn identity_follows_weights_pack_and_tensor_set() {
        let d = std::env::temp_dir().join(format!("pdk-md-{}-id", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("temp dir");
        let names = ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"];
        let owned: Vec<String> = names.iter().map(|n| n.to_string()).collect();
        let id = |fill: &[u8], names: &[String], pack: [u32; 3]| {
            let map = MappedGguf::open(&gguf(
                &d,
                &["blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"],
                fill,
            ))
            .expect("open gguf");
            expert_identity(&map, names, pack).expect("identity")
        };
        let base = id(&[1, 2], &owned, [0, 25, 0]);
        assert_eq!(base, id(&[1, 2], &owned, [0, 25, 0]), "stable");
        assert_ne!(base, id(&[1, 3], &owned, [0, 25, 0]), "weights");
        assert_ne!(
            base,
            id(&[1, 2], &owned, [0, 26, 0]),
            "pack (repack format)"
        );
        assert_ne!(base, id(&[1, 2], &owned[..1], [0, 25, 0]), "tensor set");
        let map = MappedGguf::open(&gguf(&d, &names, &[1, 2])).expect("open gguf");
        assert!(expert_identity(&map, &["blk.9.ffn_gate_exps.weight".into()], [0, 25, 0]).is_err());
    }

    #[test]
    fn repacked_sizes_follow_the_repack_rules() {
        use super::super::kquant::{kq_n_super, kq_repacked_bytes};
        let (_, _, data_b) = super::super::kq_params(GgmlType::Q4K).expect("Q4_K is k-quant");
        assert_eq!(
            kq_n_super(&[256, 640, 8], GgmlType::Q4K, "t").expect("aligned"),
            5120
        );
        assert_eq!(
            kq_repacked_bytes(&[256, 640, 8], GgmlType::Q4K, "t").expect("aligned"),
            (5120 * data_b, 5120 * 24)
        );
        // a 640-wide row is 2.5 superblocks: only the flat 32-block types
        assert!(kq_n_super(&[640, 2048, 8], GgmlType::Q4K, "t").is_err());
        assert_eq!(
            kq_n_super(&[640, 2048, 8], GgmlType::Iq4Nl, "t").expect("flat"),
            640 * 2048 * 8 / 256
        );
        assert!(
            kq_n_super(&[640, 3, 8], GgmlType::Iq4Nl, "t").is_err(),
            "partial superblock"
        );
        assert!(
            kq_repacked_bytes(&[256, 2, 4], GgmlType::F16, "t").is_err(),
            "not k-quant"
        );
    }

    #[test]
    fn slot_budget_refuses_what_cannot_serve_a_token() {
        assert_eq!(disk_slots(100 << 20, 2 << 20, None, 8, 256), Ok(50));
        assert_eq!(
            disk_slots(u64::MAX, 2 << 20, None, 8, 256),
            Ok(256),
            "capped at n_expert"
        );
        assert_eq!(disk_slots(0, 2 << 20, Some(16), 8, 256), Ok(16), "pin wins");
        assert_eq!(
            disk_slots(10 << 20, 2 << 20, None, 8, 256),
            Err(MoeDiskRefusal::Slots {
                slots: 5,
                n_active: 8
            })
        );
        assert_eq!(
            disk_slots(1 << 30, 0, None, 8, 256),
            Err(MoeDiskRefusal::Slots {
                slots: 0,
                n_active: 8
            }),
            "no price is no cache"
        );
        assert!(
            disk_slots(0, 2 << 20, Some(4), 8, 256).is_err(),
            "a pin is checked too"
        );
    }

    #[test]
    fn every_refusal_names_its_cause() {
        let msg = |r: MoeDiskRefusal| GpuError::from(r).to_string();
        assert!(msg(MoeDiskRefusal::Quant(GgmlType::Q8_0)).contains("Q8_0"));
        let m = msg(MoeDiskRefusal::Slots {
            slots: 5,
            n_active: 8,
        });
        assert!(m.contains("at least 8") && m.contains("buys 5"), "{m}");
        let m = msg(MoeDiskRefusal::Mode("CUDA graph capture"));
        assert!(
            m.contains("CUDA graph capture") && m.contains("pinned backing"),
            "{m}"
        );
        assert!(msg(MoeDiskRefusal::NoCache).contains("enable_moe_cache"));
        assert!(matches!(
            GpuError::from(MoeDiskRefusal::NoCache),
            GpuError::Unsupported(_)
        ));
    }
}
