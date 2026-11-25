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

use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::path::Path;
use std::sync::Mutex;
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

/// Routed rows one disk-backed launch serves at most: a prompt is split into
/// token ranges of at most this many `(token, expert)` pairs, which also
/// sizes each cache's routing scratch (16 KiB) instead of `max_ctx * top-k`.
pub(crate) const DISK_CHUNK_ROWS: usize = 4096;

/// Staging buffers a disk backing pins: one expert record each, so this
/// bounds the host memory the miss path holds (about 2 MB a record on the
/// 35B-A3B UD file) and the reads in flight before a fill waits.
const STAGING_RECORDS: usize = 8;

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
    /// Artifact bytes those reads returned (whole records).
    pub bytes_read: u64,
    /// Wall time in positioned reads and digest checks.
    pub read_time: Duration,
    /// Times a fill found every staging buffer in flight and waited.
    pub staging_waits: u64,
    /// Wall time the host spent waiting for slot copies to complete.
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
/// token ranges whose distinct experts fit `slots` and whose rows fit
/// `max_rows`, greedily. Every range is served by one resolve, so a range
/// never evicts what it reads.
pub(super) fn plan_chunks(
    ids: &[u32],
    n_active: usize,
    n_expert: usize,
    slots: usize,
    max_rows: usize,
    out: &mut Vec<Range<usize>>,
) -> Result<(), GpuError> {
    out.clear();
    if n_active == 0
        || slots < n_active
        || max_rows < n_active
        || !ids.len().is_multiple_of(n_active)
    {
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
        if distinct + new > slots || (t + 1 - start) * n_active > max_rows {
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

    #[cfg(test)]
    fn store(&self) -> &DiskExpertStore {
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
                return Err(GpuError::ExpertIo(e));
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

    /// Wait for the copies of earlier fills. A failed wait breaks the backing
    /// here rather than in whatever sync the caller would otherwise hit
    /// first.
    pub(super) fn settle(&mut self, dev: &mut impl SlotCopier) -> Result<(), GpuError> {
        if self.broken {
            return Err(broken());
        }
        self.drain(dev)
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
    let drv = |e: cudarc::driver::DriverError| super::from_driver(e);
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
                        .map_err(drv)?;
                    at += n;
                }
            }
            exec.stream.synchronize().map_err(drv)?;
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
    /// Per-launch host scratch: the routing, then its slot form.
    scratch: Mutex<(Vec<u32>, Vec<u32>)>,
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
            scratch: Mutex::new((Vec::new(), Vec::new())),
        })
    }

    /// Bytes one slot costs in every layer together.
    pub(crate) fn slot_price(&self) -> u64 {
        self.layouts
            .iter()
            .map(|l| l.stream_bytes().iter().sum::<usize>() as u64)
            .sum()
    }

    /// Seat `slots` per layer: the caches' LRUs and the pinned staging pool.
    /// Returns one slot cache per layer, in layer order.
    pub(crate) fn seat(
        &self,
        exec: &GpuExecutor,
        slots: usize,
        max_rows: usize,
    ) -> Result<Vec<ExpertCache>, GpuError> {
        let mut caches = Vec::with_capacity(self.layouts.len());
        for l in &self.layouts {
            caches.push(exec.new_expert_cache_for_disk(l.clone(), slots, max_rows)?);
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
        let pool = StagingPool::new(STAGING_RECORDS, len, |n| PinnedBuf::new(stream.clone(), n))
            .map_err(|e| GpuError::Driver(format!("expert staging: {e}")))?;
        *state = Some(Seat::Seated(DiskExperts::new(store, pool, slots)?));
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
    /// (a host sync) and split its tokens into ranges whose experts fit the
    /// cache. Each range is then served by [`fill_chunk`] and the caller's
    /// expert kernels over the cache's routing.
    ///
    /// [`fill_chunk`]: MoeDiskBacking::fill_chunk
    pub(crate) fn plan(
        &self,
        exec: &GpuExecutor,
        cache: &ExpertCache,
        idx: &CudaSlice<u32>,
        rows: usize,
        n_active: usize,
        chunks: &mut Vec<Range<usize>>,
    ) -> Result<(), GpuError> {
        if let Some(Seat::Seated(x)) = &mut *lock(&self.state) {
            x.settle(&mut CacheCopier { exec, cache })?;
        }
        let mut scratch = lock(&self.scratch);
        scratch.0 = exec.to_host_u32_len(idx, rows)?;
        plan_chunks(
            &scratch.0,
            n_active,
            cache.n_expert,
            cache.slots,
            cache.max_rows,
            chunks,
        )
    }

    /// Fill the experts of `tokens` (a range from the last `plan`) of
    /// `layer` into `cache` and upload those tokens' slot routing to the
    /// head of `idx_slot`, for kernels launched over just their rows.
    pub(crate) fn fill_chunk(
        &self,
        exec: &GpuExecutor,
        cache: &ExpertCache,
        layer: usize,
        n_active: usize,
        tokens: Range<usize>,
    ) -> Result<(), GpuError> {
        let mut scratch = lock(&self.scratch);
        let (ids, routing) = &mut *scratch;
        let r = tokens.start * n_active..tokens.end * n_active;
        if r.end > ids.len() {
            return Err(GpuError::Driver(format!(
                "expert disk backing: tokens {tokens:?} past the planned launch"
            )));
        }
        routing.clear();
        routing.resize(r.len(), MOE_CACHE_NONE);
        let mut state = lock(&self.state);
        let Some(Seat::Seated(x)) = &mut *state else {
            return Err(MoeDiskRefusal::NoCache.into());
        };
        x.fill(layer, &ids[r], routing, &mut CacheCopier { exec, cache })?;
        cache.upload_routing(exec, routing)
    }
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
    Seated(DiskExperts<PinnedBuf>),
}

fn broken() -> GpuError {
    GpuError::Driver(
        "expert disk backing: an earlier slot copy failed to complete; reload the model".into(),
    )
}

/// A read-only view of `len` elements of a device buffer from `off`, as
/// the whole `CudaSlice` the launch wrappers take: how a disk-backed launch
/// runs the expert kernels over one token range's rows. Never freed (the
/// parent owns the memory); it borrows the parent for as long as it lives.
pub(crate) struct RowsView<'a, T> {
    view: ManuallyDrop<CudaSlice<T>>,
    _parent: PhantomData<&'a CudaSlice<T>>,
}

/// [`RowsView`] that can be written: borrows its parent mutably.
pub(crate) struct RowsViewMut<'a, T> {
    view: ManuallyDrop<CudaSlice<T>>,
    _parent: PhantomData<&'a mut CudaSlice<T>>,
}

/// The alias both views hold. The engine runs with cudarc's event tracking
/// off, so the alias carries no events to leak.
fn alias<T>(
    exec: &GpuExecutor,
    parent: &CudaSlice<T>,
    off: usize,
    len: usize,
) -> Result<ManuallyDrop<CudaSlice<T>>, GpuError> {
    if off.checked_add(len).is_none_or(|end| end > parent.len()) {
        return Err(GpuError::Driver(format!(
            "rows view {off}+{len} over {} elements",
            parent.len()
        )));
    }
    let (p, _g) = parent.device_ptr(&exec.stream);
    let at = p + (off * std::mem::size_of::<T>()) as u64;
    // SAFETY: `at..at + len` lies inside `parent`'s live allocation (checked
    // above); the ManuallyDrop keeps the alias from ever freeing it, and the
    // views' lifetimes keep it from outliving `parent`.
    Ok(ManuallyDrop::new(unsafe {
        exec.stream.upgrade_device_ptr::<T>(at, len)
    }))
}

impl<'a, T> RowsView<'a, T> {
    pub(crate) fn new(
        exec: &GpuExecutor,
        parent: &'a CudaSlice<T>,
        off: usize,
        len: usize,
    ) -> Result<Self, GpuError> {
        Ok(Self {
            view: alias(exec, parent, off, len)?,
            _parent: PhantomData,
        })
    }
}

impl<'a, T> RowsViewMut<'a, T> {
    pub(crate) fn new(
        exec: &GpuExecutor,
        parent: &'a mut CudaSlice<T>,
        off: usize,
        len: usize,
    ) -> Result<Self, GpuError> {
        Ok(Self {
            view: alias(exec, parent, off, len)?,
            _parent: PhantomData,
        })
    }
}

impl<T> std::ops::Deref for RowsView<'_, T> {
    type Target = CudaSlice<T>;
    fn deref(&self) -> &CudaSlice<T> {
        &self.view
    }
}

impl<T> std::ops::Deref for RowsViewMut<'_, T> {
    type Target = CudaSlice<T>;
    fn deref(&self) -> &CudaSlice<T> {
        &self.view
    }
}

impl<T> std::ops::DerefMut for RowsViewMut<'_, T> {
    fn deref_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.view
    }
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
        plan_chunks(&[1, 2], 2, N_EXPERT, 4, 64, &mut out).expect("plan");
        assert_eq!(out, vec![0..1]);
        // tokens 0-1 touch {0,1,2,3}; token 2 adds 4 -> new range
        plan_chunks(&[0, 1, 2, 3, 4, 0, 4, 5], 2, N_EXPERT, 4, 64, &mut out).expect("plan");
        assert_eq!(out, vec![0..2, 2..4]);
        // repeated experts do not count twice
        plan_chunks(&[1, 1, 1, 2, 2, 1], 2, N_EXPERT, 2, 64, &mut out).expect("plan");
        assert_eq!(out, vec![0..3]);
        assert!(
            plan_chunks(&[0, 8], 2, N_EXPERT, 4, 64, &mut out).is_err(),
            "id range"
        );
        assert!(
            plan_chunks(&[0, 1], 2, N_EXPERT, 1, 64, &mut out).is_err(),
            "slots < top-k"
        );
        assert!(
            plan_chunks(&[0, 1, 2], 2, N_EXPERT, 4, 64, &mut out).is_err(),
            "ragged"
        );
        // the row bound splits even when the experts would fit
        plan_chunks(&[1, 2, 1, 2, 1, 2], 2, N_EXPERT, 4, 4, &mut out).expect("plan");
        assert_eq!(out, vec![0..2, 2..3]);
        assert!(
            plan_chunks(&[1, 2], 2, N_EXPERT, 4, 1, &mut out).is_err(),
            "rows < top-k"
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
