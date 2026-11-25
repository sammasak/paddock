//! Device-mapped host mirrors of repacked weight planes - where a MoE
//! model's routed experts live under `[moe_offload]`.
//!
//! The plane sits in page-locked host memory the GPU can address directly
//! (`cuMemHostAlloc` with `DEVICEMAP`), so the MoE kernels read it over
//! PCIe on the same `(e*ff + o)` addressing they use for a resident plane:
//! nothing on the launch side changes and a captured decode graph stays
//! valid. The mirror is built from the same repack the resident path runs -
//! staged, repacked on the GPU, copied down, device copy freed - so it is
//! byte-identical to what the resident plane would hold and greedy parity
//! against it is a bit-identity check (tests/gpu_kquant_parity.rs).
//!
//! Reading every expert over the bus is the floor (~9 tok/s on a PCIe 4.0
//! x8 35B-A3B); `moe_cache.rs` keeps the hot ones in VRAM on top of this.
//!
//! Windows bounds device-mapped pinned memory by the OS's lockable-page
//! budget (measured on a 64 GB box: 20 GB allocates, 28 GB is refused). A
//! refusal is a load error naming the size, never a pageable fallback - the
//! kernels cannot read pageable memory.

use std::mem::ManuallyDrop;
use std::sync::Arc;

use cudarc::driver::sys as cu;
use cudarc::driver::{CudaStream, DevicePtr};

use super::{GpuError, GpuExecutor, RepackedKQ};
use paddock_models::mapped::MappedGguf;

/// One device-mapped pinned host extent, RAII over `cuMemHostAlloc` /
/// `cuMemFreeHost`.
pub struct HostMirror {
    host: *mut core::ffi::c_void,
    dev: cu::CUdeviceptr,
    len: usize,
    /// The stream the plane is read on - drained before the pages go away.
    stream: Arc<CudaStream>,
}

// SAFETY: the pointers are owned here, the extent is written once at load
// (before any kernel reads it) and read-only afterwards, and the engine
// thread owns every use. Send is needed because the model that owns the
// plane moves to the engine thread after load.
unsafe impl Send for HostMirror {}
unsafe impl Sync for HostMirror {}

impl HostMirror {
    /// Allocate `len` bytes of device-mapped pinned memory.
    fn new(stream: Arc<CudaStream>, len: usize, what: &str) -> Result<Self, GpuError> {
        let mut host: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: plain driver call; the out-param is written on success.
        let rc = unsafe {
            cu::cuMemHostAlloc(
                &mut host,
                len,
                cu::CU_MEMHOSTALLOC_PORTABLE | cu::CU_MEMHOSTALLOC_DEVICEMAP,
            )
        };
        if rc != cu::CUresult::CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuMemHostAlloc({} MiB, PORTABLE|DEVICEMAP) for {what}: {rc:?} - the OS refused \
                 that much page-locked memory; lower the offloaded expert bytes or free RAM",
                len >> 20
            )));
        }
        let mut dev: cu::CUdeviceptr = 0;
        // SAFETY: `host` is a live DEVICEMAP allocation from the call above.
        let rc = unsafe { cu::cuMemHostGetDevicePointer_v2(&mut dev, host, 0) };
        if rc != cu::CUresult::CUDA_SUCCESS {
            // SAFETY: freeing what we just allocated, exactly once.
            let _ = unsafe { cu::cuMemFreeHost(host) };
            return Err(GpuError::Driver(format!(
                "cuMemHostGetDevicePointer for {what}: {rc:?}"
            )));
        }
        Ok(Self {
            host,
            dev,
            len,
            stream,
        })
    }

    /// Fill the mirror from a device buffer of exactly `len` bytes.
    fn fill_from_device(&mut self, src: cu::CUdeviceptr) -> Result<(), GpuError> {
        self.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("host mirror: pre-copy sync: {e}")))?;
        // SAFETY: `host` spans `len` bytes and nothing reads it yet.
        let dst = unsafe { core::slice::from_raw_parts_mut(self.host.cast::<u8>(), self.len) };
        // SAFETY: `src` is a live device buffer of at least `len` bytes.
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(dst, src) }
            .map_err(|e| GpuError::Driver(format!("host mirror: dtoh: {e}")))
    }

    /// A `CudaSlice` view over the mirror's device address. The slice is
    /// returned under `ManuallyDrop`: its `Drop` would `cuMemFree` a pointer
    /// that came from `cuMemHostAlloc`. The mirror owns the pages.
    fn device_view(&self) -> ManuallyDrop<cudarc::driver::CudaSlice<u8>> {
        // SAFETY: `dev` addresses `len` bytes for as long as `self` lives, and
        // the ManuallyDrop keeps the slice from ever freeing it.
        ManuallyDrop::new(unsafe { self.stream.upgrade_device_ptr::<u8>(self.dev, self.len) })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: `host` spans `len` bytes, written once at load and
        // read-only afterwards (the device only reads it too).
        unsafe { core::slice::from_raw_parts(self.host.cast::<u8>(), self.len) }
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for HostMirror {
    fn drop(&mut self) {
        // The plane may still be read by in-flight work on the stream.
        let _ = self.stream.synchronize();
        // SAFETY: `host` came from cuMemHostAlloc and is freed exactly once.
        let _ = unsafe { cu::cuMemFreeHost(self.host) };
    }
}

/// Page-locked host bytes for the disk backing's staging pool
/// (`moe_disk.rs`): a miss is read into one, then copied to its slot with an
/// async copy, which only overlaps the host when the source is pinned.
/// Not device-mapped - nothing on the device addresses it directly.
pub struct PinnedBuf {
    ptr: *mut u8,
    len: usize,
    /// The stream copies out of this buffer run on - drained before the
    /// pages go away.
    stream: Arc<CudaStream>,
}

// SAFETY: the pointer is a driver allocation this value alone owns; the
// engine thread owns every use, as for HostMirror.
unsafe impl Send for PinnedBuf {}

impl PinnedBuf {
    pub(super) fn new(stream: Arc<CudaStream>, len: usize) -> std::io::Result<Self> {
        // SAFETY: plain driver call; the pointer is owned here until Drop.
        let ptr = unsafe { cudarc::driver::result::malloc_host(len, cu::CU_MEMHOSTALLOC_PORTABLE) }
            .map_err(|e| {
                std::io::Error::other(format!(
                    "cuMemHostAlloc({} MiB) for expert staging: {e}",
                    len >> 20
                ))
            })?;
        Ok(Self {
            ptr: ptr.cast(),
            len,
            stream,
        })
    }
}

impl AsRef<[u8]> for PinnedBuf {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: `ptr` spans `len` bytes for as long as `self` lives.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl AsMut<[u8]> for PinnedBuf {
    fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above; `&mut self` makes the host view exclusive. The
        // staging pool hands out a buffer only after the copies reading it
        // completed.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        // A copy may still be reading the pages.
        let _ = self.stream.synchronize();
        // SAFETY: ptr came from malloc_host and is freed exactly once.
        let _ = unsafe { cudarc::driver::result::free_host(self.ptr.cast()) };
    }
}

/// A repacked k-quant plane whose bytes live in host-mapped memory. Derefs
/// to the same `RepackedKQ` the kernels take, so every launch wrapper is
/// unchanged; the two `CudaSlice`s inside are non-owning views (see
/// [`HostMirror::device_view`]) and the mirrors own the pages.
pub struct HostMappedKq {
    plane: ManuallyDrop<RepackedKQ>,
    // Declared after `plane` so the views are forgotten before the pages
    // they point at are released.
    _data: HostMirror,
    _scales: HostMirror,
}

impl HostMappedKq {
    /// Bytes held in host memory (data + scale streams).
    pub fn host_bytes(&self) -> u64 {
        (self._data.len() + self._scales.len()) as u64
    }

    /// The data and scale streams as host bytes - what a disk artifact is
    /// written from.
    pub(super) fn host_streams(&self) -> (&[u8], &[u8]) {
        (self._data.as_slice(), self._scales.as_slice())
    }
}

impl std::ops::Deref for HostMappedKq {
    type Target = RepackedKQ;
    fn deref(&self) -> &RepackedKQ {
        &self.plane
    }
}

impl GpuExecutor {
    /// [`GpuExecutor::try_repack_kquant`], landing in a host mirror instead
    /// of VRAM: repack on the GPU as usual, copy the two streams down, free
    /// the device copy. `None` when the tensor is not a k-quant type (the
    /// caller falls through to its Q8 path exactly as it does today).
    pub fn try_repack_kquant_host_mapped(
        &self,
        map: &MappedGguf,
        name: &str,
    ) -> Result<Option<HostMappedKq>, GpuError> {
        let Some(dev) = self.try_repack_kquant(map, name)? else {
            return Ok(None);
        };
        let RepackedKQ {
            data,
            scales,
            dims,
            ty,
            act128,
        } = dev;
        let mut data_m = HostMirror::new(self.stream.clone(), data.len(), name)?;
        {
            let (p, _g) = data.device_ptr(&self.stream);
            data_m.fill_from_device(p)?;
        }
        drop(data);
        let mut scales_m = HostMirror::new(self.stream.clone(), scales.len(), name)?;
        {
            let (p, _g) = scales.device_ptr(&self.stream);
            scales_m.fill_from_device(p)?;
        }
        drop(scales);
        let plane = RepackedKQ {
            data: ManuallyDrop::into_inner(data_m.device_view()),
            scales: ManuallyDrop::into_inner(scales_m.device_view()),
            dims,
            ty,
            act128,
        };
        Ok(Some(HostMappedKq {
            plane: ManuallyDrop::new(plane),
            _data: data_m,
            _scales: scales_m,
        }))
    }
}
