//! MoE expert offload, the VRAM side: a per-layer LRU cache of routed
//! experts over the host-mapped planes of `host_plane.rs`.
//!
//! The cache is three slot planes (gate/up/down) in the same repacked
//! k-quant layout as a resident plane, holding `slots` experts instead of
//! `n_expert`, plus the device-side bookkeeping the pack's resolve kernel
//! updates in place: `slot_of[n_expert]`, `expert_in[slots]`,
//! `last_use[slots]`, a tick counter. Per launch the engine runs
//! resolve (ids -> slots, miss jobs) then fill (miss bytes from the mirror
//! into their slots, PCIe-bound) and hands the unchanged MoE kernels the
//! slot planes and the remapped ids. All of it captures into the decode
//! graph; nothing here touches the host after load. The disk backing
//! (`moe_disk.rs`) seats the same slot planes without fill sources and
//! fills them from the host through [`CacheCopier`].
//!
//! Sizing: a slot is one expert's gate+up+down bytes (2.0 MB on the 35B-A3B
//! UD file). Measured on a 4032-token prompt, per-layer LRU hit rates run
//! 82% at 64 slots, 90% at 96, 94% at 128; enable_batch seats what the KV
//! plan leaves.

use cudarc::driver::{CudaSlice, DevicePtr};
use paddock_models::ggml_type::GgmlType;

use super::error::check;
use super::{GpuError, GpuExecutor, HostMappedKq, RepackedKQ};

/// Sentinel in `slot_of` / `expert_in`: not resident.
pub const MOE_CACHE_NONE: u32 = u32::MAX;

/// The `[moe_offload]` election, armed once by the runner before any model
/// loads (families read it at load and at enable_batch) - the same shape as
/// `kv_tier::pool_tier::set_tier_ram_bytes`. Budgets only: `vram_gb` caps
/// the slot cache; `None` lets the cache take what the KV plan leaves.
#[derive(Clone, Copy, Debug, Default)]
pub struct MoeOffloadCfg {
    pub enabled: bool,
    pub vram_bytes: Option<u64>,
}

static MOE_OFFLOAD: std::sync::OnceLock<MoeOffloadCfg> = std::sync::OnceLock::new();

pub fn set_moe_offload(cfg: MoeOffloadCfg) {
    let _ = MOE_OFFLOAD.set(cfg);
}

/// The armed config; `PADDOCK_MOE_HOST=1` is the development switch that
/// enables it without a config file (tests, bring-up), and
/// `PADDOCK_MOE_DISK` (see [`moe_disk_artifact`]) implies it.
pub fn moe_offload() -> MoeOffloadCfg {
    let mut c = MOE_OFFLOAD.get().copied().unwrap_or_default();
    if paddock_models::dev_var_os!("PADDOCK_MOE_HOST").is_some() || moe_disk_artifact().is_some() {
        c.enabled = true;
    }
    c
}

/// `PADDOCK_MOE_DISK=<artifact>`: development switch for the disk backing -
/// the routed experts are read on a miss from this artifact (`moe_disk.rs`)
/// instead of being held in page-locked RAM. qwen35 serves it, qwen4exp
/// refuses it at load, and families without `[moe_offload]` ignore it as
/// they ignore `PADDOCK_MOE_HOST`.
pub fn moe_disk_artifact() -> Option<std::path::PathBuf> {
    paddock_models::dev_var_os!("PADDOCK_MOE_DISK")
        .filter(|v| !v.is_empty())
        .map(Into::into)
}

/// `PADDOCK_MOE_CACHE_SLOTS=<n>`: development pin for the per-layer slot
/// count, overriding the auto size (the same instrument class as
/// PADDOCK_KV_POOL_BLOCKS). 0 = no cache (pure zero-copy).
pub fn moe_cache_slots_pin() -> Option<usize> {
    paddock_models::dev_var!("PADDOCK_MOE_CACHE_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
}

/// Geometry of a repacked expert plane, independent of its backing address.
/// Cache slots need this layout, not ownership of the original weight bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExpertPlaneLayout {
    pub(super) dims: Vec<usize>,
    pub(super) ty: GgmlType,
    pub(super) data_per_expert: usize,
    pub(super) scales_per_expert: usize,
}

impl ExpertPlaneLayout {
    pub(super) fn new(
        dims: &[usize],
        ty: GgmlType,
        data_bytes: usize,
        scales_bytes: usize,
    ) -> Result<Self, GpuError> {
        let n_expert = match dims {
            [_, _, n] if *n > 0 => *n,
            _ => {
                return Err(GpuError::Driver(
                    "expert cache: invalid plane geometry".into(),
                ));
            }
        };
        if data_bytes == 0
            || scales_bytes == 0
            || !data_bytes.is_multiple_of(n_expert)
            || !scales_bytes.is_multiple_of(n_expert)
        {
            return Err(GpuError::Driver(
                "expert cache: plane bytes are not a whole number of experts".into(),
            ));
        }
        Ok(Self {
            dims: dims.to_vec(),
            ty,
            data_per_expert: data_bytes / n_expert,
            scales_per_expert: scales_bytes / n_expert,
        })
    }

    fn from_host(p: &HostMappedKq) -> Result<Self, GpuError> {
        Self::new(&p.dims, p.ty, p.data.len(), p.scales.len())
    }
}

/// One MoE layer's gate/up/down plane geometry: what a slot holds and what
/// every backing (pinned mirror or disk artifact) must agree on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExpertLayerLayout {
    pub(super) planes: [ExpertPlaneLayout; 3],
}

impl ExpertLayerLayout {
    pub(crate) fn new(planes: [ExpertPlaneLayout; 3]) -> Result<Self, GpuError> {
        let n_expert = planes[0].dims[2];
        if planes[1].dims[2] != n_expert || planes[2].dims[2] != n_expert {
            return Err(GpuError::Driver(format!(
                "expert cache: gate {n_expert} experts, up {}, down {}",
                planes[1].dims[2], planes[2].dims[2]
            )));
        }
        Ok(Self { planes })
    }

    pub(super) fn from_pinned(
        gate: &HostMappedKq,
        up: &HostMappedKq,
        down: &HostMappedKq,
    ) -> Result<Self, GpuError> {
        Self::new([
            ExpertPlaneLayout::from_host(gate)?,
            ExpertPlaneLayout::from_host(up)?,
            ExpertPlaneLayout::from_host(down)?,
        ])
    }

    pub(super) fn n_expert(&self) -> usize {
        self.planes[0].dims[2]
    }

    /// Bytes per expert of the six streams, in fill-descriptor order: gate
    /// data, gate scales, up data, up scales, down data, down scales.
    pub(super) fn stream_bytes(&self) -> [usize; 6] {
        let mut b = [0usize; 6];
        for (i, p) in self.planes.iter().enumerate() {
            b[2 * i] = p.data_per_expert;
            b[2 * i + 1] = p.scales_per_expert;
        }
        b
    }
}

pub struct ExpertCache {
    pub slots: usize,
    pub n_expert: usize,
    /// Rows the per-launch scratch can take (`idx_slot`, `jobs`).
    pub max_rows: usize,
    pub gate: RepackedKQ,
    pub up: RepackedKQ,
    pub down: RepackedKQ,
    slot_of: CudaSlice<u32>,
    expert_in: CudaSlice<u32>,
    last_use: CudaSlice<u32>,
    tick: CudaSlice<u32>,
    idx_slot: CudaSlice<u32>,
    jobs: CudaSlice<u32>,
    n_jobs: CudaSlice<u32>,
    /// `[rows resolved, misses]`, accumulated by every resolve.
    stats: CudaSlice<u32>,
    /// Expert-major prefill (waves): a prefill launch whose routed rows
    /// exceed `slots` is served in `n_waves` (= ceil(n_expert / slots))
    /// passes over the token-batched pair, each pass marking every
    /// out-of-wave pair ABSENT (`MOE_CACHE_NONE` in the routing, which the
    /// pair kernels skip). See `pd_moe_wave_plan` in the pack.
    pub n_waves: usize,
    /// `wave_of[n_expert]`: the wave each present expert landed in (none
    /// when absent from the launch), written by the plan.
    wave_of: CudaSlice<u32>,
    /// `wave_ids[n_waves * slots]`, `wave_cnt[n_waves]`: the plan's lists.
    wave_ids: CudaSlice<u32>,
    wave_cnt: CudaSlice<u32>,
    /// The wave's remapped routing (`max_rows`), written by the mask.
    idx_wave: CudaSlice<u32>,
    /// The wave's compacted in-wave pair indices (`max_rows`) + device
    /// count, and the tokens holding at least one (`max_rows`, over-sized)
    /// + count - what the LIST pair kernels stride over.
    pairs_w: CudaSlice<u32>,
    n_pairs_w: CudaSlice<u32>,
    rows_w: CudaSlice<u32>,
    n_rows_w: CudaSlice<u32>,
    /// Fill descriptors: mirror sources, slot destinations, bytes per expert -
    /// gate data, gate scales, up data, up scales, down data, down scales.
    /// No sources on a disk-backed cache: its misses arrive through
    /// [`CacheCopier`], never the fill kernel.
    src: Option<[u64; 6]>,
    dst: [u64; 6],
    bytes: [u64; 6],
}

impl ExpertCache {
    /// Remapped routing (slot ids) written by the last resolve; the MoE
    /// kernels take it in place of the expert ids.
    pub fn idx_slot(&self) -> &CudaSlice<u32> {
        &self.idx_slot
    }

    /// The routing the wave mask wrote (the MoE pair takes it like
    /// `idx_slot` on the wave path).
    pub fn idx_wave(&self) -> &CudaSlice<u32> {
        &self.idx_wave
    }

    /// The wave's pair list + count and token list + count (see the mask).
    pub fn wave_lists(
        &self,
    ) -> (
        &CudaSlice<u32>,
        &CudaSlice<u32>,
        &CudaSlice<u32>,
        &CudaSlice<u32>,
    ) {
        (&self.pairs_w, &self.n_pairs_w, &self.rows_w, &self.n_rows_w)
    }

    /// `(rows resolved, misses)` since load - a sync + tiny readback, for
    /// logs and gates, never on the tick.
    pub fn stats(&self, exec: &GpuExecutor) -> Result<(u64, u64), GpuError> {
        let v = exec.to_host_u32(&self.stats)?;
        Ok((v[0] as u64, v[1] as u64))
    }

    /// VRAM the slot planes hold.
    pub fn vram_bytes(&self) -> u64 {
        (self.gate.data.len()
            + self.gate.scales.len()
            + self.up.data.len()
            + self.up.scales.len()
            + self.down.data.len()
            + self.down.scales.len()) as u64
    }

    /// Bytes one slot (one expert across the three planes) costs, from the
    /// mirrors' per-expert strides.
    pub fn slot_bytes(gate: &HostMappedKq, up: &HostMappedKq, down: &HostMappedKq) -> u64 {
        let per = |p: &RepackedKQ| ((p.data.len() + p.scales.len()) / p.dims[2]) as u64;
        per(gate) + per(up) + per(down)
    }
}

impl GpuExecutor {
    pub fn has_moe_cache(&self) -> bool {
        self.kernels.moe_cache_resolve.is_some() && self.kernels.moe_cache_fill.is_some()
    }

    /// The expert-major prefill path (slots 580-582) is in the pack.
    pub fn has_moe_wave(&self) -> bool {
        self.kernels.moe_wave_plan.is_some()
            && self.kernels.moe_cache_resolve_dev.is_some()
            && self.kernels.moe_wave_mask.is_some()
            && self.kernels.kquant_moe_gate_up_list.is_some()
            && self.kernels.kquant_moe_down_list.is_some()
    }

    /// Build a `slots`-expert cache over three host-mapped planes of one
    /// layer. Slot planes are allocated empty; the first ticks fill them.
    pub fn new_expert_cache(
        &self,
        gate: &HostMappedKq,
        up: &HostMappedKq,
        down: &HostMappedKq,
        slots: usize,
        max_rows: usize,
    ) -> Result<ExpertCache, GpuError> {
        let layout = ExpertLayerLayout::from_pinned(gate, up, down)?;
        let mut src = [0u64; 6];
        for (i, p) in [gate, up, down].into_iter().enumerate() {
            let (sp, _g1) = p.data.device_ptr(&self.stream);
            let (ssp, _g2) = p.scales.device_ptr(&self.stream);
            src[2 * i] = sp;
            src[2 * i + 1] = ssp;
        }
        self.new_expert_cache_from_layout(layout, Some(src), slots, max_rows)
    }

    /// A `slots`-expert cache whose misses the disk backing fills from the
    /// host (`moe_disk.rs`): same slot planes, no device-side fill sources.
    /// Its launches route through the backing's shared buffers, so the
    /// cache's own routing scratch is one row.
    pub(super) fn new_expert_cache_for_disk(
        &self,
        layout: ExpertLayerLayout,
        slots: usize,
    ) -> Result<ExpertCache, GpuError> {
        self.new_expert_cache_from_layout(layout, None, slots, 1)
    }

    // Keep the existing pinned-host fill descriptors and kernel ABI intact.
    // The cache allocation itself depends only on geometry and slot count.
    fn new_expert_cache_from_layout(
        &self,
        layout: ExpertLayerLayout,
        src: Option<[u64; 6]>,
        slots: usize,
        max_rows: usize,
    ) -> Result<ExpertCache, GpuError> {
        let n_expert = layout.n_expert();
        if slots == 0 || slots > n_expert {
            return Err(GpuError::Driver(format!(
                "expert cache: {slots} slots over {n_expert} experts"
            )));
        }
        let mut dst = [0u64; 6];
        let mut bytes = [0u64; 6];
        let mut planes = Vec::with_capacity(3);
        for (i, p) in layout.planes.into_iter().enumerate() {
            let per_data = p.data_per_expert;
            let per_scales = p.scales_per_expert;
            let data = self.alloc_u8(per_data * slots)?;
            let scales = self.alloc_u8(per_scales * slots)?;
            {
                let (dp, _g3) = data.device_ptr(&self.stream);
                let (dsp, _g4) = scales.device_ptr(&self.stream);
                dst[2 * i] = dp;
                dst[2 * i + 1] = dsp;
            }
            bytes[2 * i] = per_data as u64;
            bytes[2 * i + 1] = per_scales as u64;
            let mut dims = p.dims;
            dims[2] = slots;
            planes.push(RepackedKQ {
                data,
                scales,
                dims,
                ty: p.ty,
                act128: false,
            });
        }
        let n_waves = n_expert.div_ceil(slots);
        let down_p = planes.pop().expect("three planes pushed");
        let up_p = planes.pop().expect("three planes pushed");
        let gate_p = planes.pop().expect("three planes pushed");
        Ok(ExpertCache {
            slots,
            n_expert,
            max_rows,
            gate: gate_p,
            up: up_p,
            down: down_p,
            slot_of: self.to_device_u32(&vec![MOE_CACHE_NONE; n_expert])?,
            expert_in: self.to_device_u32(&vec![MOE_CACHE_NONE; slots])?,
            last_use: self.to_device_u32(&vec![0u32; slots])?,
            tick: self.to_device_u32(&[0u32])?,
            idx_slot: self.to_device_u32(&vec![0u32; max_rows])?,
            jobs: self.to_device_u32(&vec![0u32; 2 * max_rows])?,
            n_jobs: self.to_device_u32(&[0u32])?,
            stats: self.to_device_u32(&[0u32, 0u32])?,
            n_waves,
            wave_of: self.to_device_u32(&vec![MOE_CACHE_NONE; n_expert])?,
            wave_ids: self.to_device_u32(&vec![0u32; n_waves * slots])?,
            wave_cnt: self.to_device_u32(&vec![0u32; n_waves])?,
            idx_wave: self.to_device_u32(&vec![0u32; max_rows])?,
            pairs_w: self.to_device_u32(&vec![0u32; max_rows])?,
            n_pairs_w: self.to_device_u32(&[0u32])?,
            rows_w: self.to_device_u32(&vec![0u32; max_rows])?,
            n_rows_w: self.to_device_u32(&[0u32])?,
            src,
            dst,
            bytes,
        })
    }

    /// Resolve `rows` routed ids (`idx`) against the cache: writes
    /// `idx_slot`, updates the LRU state, records miss jobs. `rows` must not
    /// exceed the cache's slots (a tick never evicts what it reads).
    pub fn moe_cache_resolve(
        &self,
        c: &ExpertCache,
        idx: &CudaSlice<u32>,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_cache_resolve
            .ok_or(GpuError::MissingOp("moe_cache_resolve"))?;
        if rows > c.slots || rows > c.max_rows {
            return Err(GpuError::Driver(format!(
                "expert cache resolve: {rows} rows over {} slots / {} scratch rows",
                c.slots, c.max_rows
            )));
        }
        let (ip, _g0) = idx.device_ptr(&self.stream);
        let (so, _g1) = c.slot_of.device_ptr(&self.stream);
        let (ei, _g2) = c.expert_in.device_ptr(&self.stream);
        let (lu, _g3) = c.last_use.device_ptr(&self.stream);
        let (tk, _g4) = c.tick.device_ptr(&self.stream);
        let (is, _g5) = c.idx_slot.device_ptr(&self.stream);
        let (jb, _g6) = c.jobs.device_ptr(&self.stream);
        let (nj, _g7) = c.n_jobs.device_ptr(&self.stream);
        let (st, _g8) = c.stats.device_ptr(&self.stream);
        // SAFETY: pack ABI v1 contract; the state buffers are written by the
        // kernel in stream order and read by nothing else off-stream.
        check(unsafe {
            f(
                ip as *const _,
                rows as u32,
                c.slots as u32,
                so as *mut _,
                ei as *mut _,
                lu as *mut _,
                tk as *mut _,
                is as *mut _,
                jb as *mut _,
                nj as *mut _,
                st as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Expert-major prefill, step 1: plan the waves for `rows` routed ids.
    pub fn moe_wave_plan(
        &self,
        c: &ExpertCache,
        idx: &CudaSlice<u32>,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_wave_plan
            .ok_or(GpuError::MissingOp("moe_wave_plan"))?;
        if rows > c.max_rows {
            return Err(GpuError::Driver(format!(
                "expert wave plan: {rows} rows over {} scratch rows",
                c.max_rows
            )));
        }
        let (ip, _g0) = idx.device_ptr(&self.stream);
        let (wo, _g1) = c.wave_of.device_ptr(&self.stream);
        let (wi, _g2) = c.wave_ids.device_ptr(&self.stream);
        let (wc, _g3) = c.wave_cnt.device_ptr(&self.stream);
        // SAFETY: pack ABI v1 contract (slot 580); buffers sized at creation
        check(unsafe {
            f(
                ip as *const _,
                rows as u32,
                c.n_expert as u32,
                c.slots as u32,
                c.n_waves as u32,
                wo as *mut _,
                wi as *mut _,
                wc as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Expert-major prefill, step 2 (per wave): resolve the wave's ids
    /// through the LRU and record the miss jobs - `moe_cache_fill(c,
    /// c.slots)` copies them.
    pub fn moe_wave_resolve(&self, c: &ExpertCache, wave: usize) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_cache_resolve_dev
            .ok_or(GpuError::MissingOp("moe_cache_resolve_dev"))?;
        let (wi, _g0) = c.wave_ids.device_ptr(&self.stream);
        let (wc, _g1) = c.wave_cnt.device_ptr(&self.stream);
        let (so, _g2) = c.slot_of.device_ptr(&self.stream);
        let (ei, _g3) = c.expert_in.device_ptr(&self.stream);
        let (lu, _g4) = c.last_use.device_ptr(&self.stream);
        let (tk, _g5) = c.tick.device_ptr(&self.stream);
        let (jb, _g6) = c.jobs.device_ptr(&self.stream);
        let (nj, _g7) = c.n_jobs.device_ptr(&self.stream);
        let (st, _g8) = c.stats.device_ptr(&self.stream);
        // SAFETY: pack ABI v1 contract (slot 581); the id list and its count
        // are the plan's device outputs, offset to this wave
        check(unsafe {
            f(
                (wi as usize + wave * c.slots * 4) as *const _,
                (wc as usize + wave * 4) as *const _,
                c.slots as u32,
                so as *mut _,
                ei as *mut _,
                lu as *mut _,
                tk as *mut _,
                jb as *mut _,
                nj as *mut _,
                st as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Expert-major prefill, step 3 (per wave): the wave's routing into
    /// `idx_wave` - in-wave pairs to their slots, the rest marked absent.
    pub fn moe_wave_mask(
        &self,
        c: &ExpertCache,
        idx: &CudaSlice<u32>,
        rows: usize,
        n_active: usize,
        wave: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_wave_mask
            .ok_or(GpuError::MissingOp("moe_wave_mask"))?;
        let (ip, _g0) = idx.device_ptr(&self.stream);
        let (wo, _g1) = c.wave_of.device_ptr(&self.stream);
        let (so, _g2) = c.slot_of.device_ptr(&self.stream);
        let (iw, _g3) = c.idx_wave.device_ptr(&self.stream);
        let (pw, _g4) = c.pairs_w.device_ptr(&self.stream);
        let (npw, _g5) = c.n_pairs_w.device_ptr(&self.stream);
        let (rw, _g6) = c.rows_w.device_ptr(&self.stream);
        let (nrw, _g7) = c.n_rows_w.device_ptr(&self.stream);
        // SAFETY: pack ABI v1 contract (slot 582); rows <= max_rows checked by the plan
        check(unsafe {
            f(
                ip as *const _,
                rows as u32,
                n_active as u32,
                wo as *const _,
                so as *const _,
                wave as u32,
                MOE_CACHE_NONE,
                iw as *mut _,
                pw as *mut _,
                npw as *mut _,
                rw as *mut _,
                nrw as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Copy the last resolve's misses (at most `rows` of them) from the host
    /// mirror into their slots.
    pub fn moe_cache_fill(&self, c: &ExpertCache, rows: usize) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_cache_fill
            .ok_or(GpuError::MissingOp("moe_cache_fill"))?;
        let src = c.src.as_ref().ok_or_else(|| {
            GpuError::Driver("expert cache: a disk-backed cache has no device fill sources".into())
        })?;
        let (jb, _g1) = c.jobs.device_ptr(&self.stream);
        let (nj, _g2) = c.n_jobs.device_ptr(&self.stream);
        // SAFETY: pack ABI v1 contract; src/dst/bytes are host arrays the
        // launcher copies by value before returning.
        check(unsafe {
            f(
                jb as *const _,
                nj as *const _,
                rows as u32,
                src.as_ptr() as *const _,
                c.dst.as_ptr() as *const _,
                c.bytes.as_ptr() as *const _,
                self.stream_ptr(),
            )
        })
    }
}

/// The disk backing's device half (`moe_disk::SlotCopier`) for one layer's
/// cache: plain host-to-device copies into the slot planes on the engine
/// stream, so the expert kernels launched after them see the bytes.
pub(super) struct CacheCopier<'a> {
    pub(super) exec: &'a GpuExecutor,
    pub(super) cache: &'a ExpertCache,
}

impl super::moe_disk::SlotCopier for CacheCopier<'_> {
    type Mark = cudarc::driver::CudaEvent;

    unsafe fn enqueue(&mut self, stream: usize, slot: u32, src: &[u8]) -> Result<(), GpuError> {
        let c = self.cache;
        if stream >= 6 || slot as usize >= c.slots || src.len() as u64 != c.bytes[stream] {
            return Err(GpuError::Driver(format!(
                "expert cache: copy of {} bytes into stream {stream} slot {slot} of {}",
                src.len(),
                c.slots
            )));
        }
        let dst = c.dst[stream] + slot as u64 * c.bytes[stream];
        // SAFETY: `dst..dst + len` lies inside the slot plane (checked
        // above); the caller keeps `src` alive and unmodified until a wait on
        // a later mark.
        unsafe { cudarc::driver::result::memcpy_htod_async(dst, src, self.exec.stream.cu_stream()) }
            .map_err(super::error::from_driver)
    }

    fn mark(&mut self) -> Result<Self::Mark, GpuError> {
        self.exec.record_event()
    }

    fn wait(&mut self, mark: &Self::Mark) -> Result<(), GpuError> {
        mark.synchronize().map_err(super::error::from_driver)
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn expert_plane_layout_does_not_depend_on_a_device_pointer() {
        let layout = ExpertPlaneLayout::new(&[256, 640, 8], GgmlType::Q4K, 4096, 768)
            .expect("valid expert plane");
        assert_eq!(layout.dims, vec![256, 640, 8]);
        assert_eq!(layout.data_per_expert, 512);
        assert_eq!(layout.scales_per_expert, 96);
    }

    #[test]
    fn expert_plane_layout_rejects_invalid_geometry_and_strides() {
        for (dims, data, scales) in [
            (&[256, 640][..], 4096, 768),
            (&[256, 640, 0][..], 4096, 768),
            (&[256, 640, 8][..], 4097, 768),
            (&[256, 640, 8][..], 4096, 769),
            (&[256, 640, 8][..], 0, 768),
        ] {
            assert!(ExpertPlaneLayout::new(dims, GgmlType::Q4K, data, scales).is_err());
        }
    }

    #[test]
    fn expert_layer_layout_orders_streams_like_the_fill_descriptors() {
        let p = |rows, data, scales| {
            ExpertPlaneLayout::new(&[256, rows, 8], GgmlType::Q4K, data, scales)
                .expect("valid expert plane")
        };
        let l = ExpertLayerLayout::new([p(640, 4096, 768), p(640, 8192, 1536), p(320, 2048, 384)])
            .expect("valid layer");
        assert_eq!(l.n_expert(), 8);
        assert_eq!(l.stream_bytes(), [512, 96, 1024, 192, 256, 48]);
        let other = ExpertPlaneLayout::new(&[256, 640, 4], GgmlType::Q4K, 4096, 768)
            .expect("valid expert plane");
        assert!(ExpertLayerLayout::new([p(640, 4096, 768), other, p(320, 2048, 384)]).is_err());
    }
}
