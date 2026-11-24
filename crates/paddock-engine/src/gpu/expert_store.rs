//! MoE expert offload, the storage side of the disk backing: an immutable
//! artifact of repacked expert planes that a cache miss reads from, and the
//! bounded staging pool those reads land in.
//!
//! The pinned backing (`host_plane.rs`) needs none of this: its planes are
//! device-addressable and the fill kernel reads them in place. A disk
//! artifact is not a device pointer, so a miss is a positioned read into a
//! staging buffer and then a copy into the slot (`moe_cache.rs`). Both
//! backings describe themselves with the same `ExpertLayerLayout`; a disk
//! artifact whose layout differs from the model's is refused at open.
//!
//! One file per model, little-endian:
//!
//! - header: magic `PDXA`, format version, the caller's 32-byte identity
//!   (what the artifact was produced from: model digest and repack format),
//!   layer count, expert count, then per layer the three plane geometries
//!   (ggml type, dims, data and scale bytes per expert);
//! - a BLAKE3 digest per `(layer, expert)` record;
//! - the records, layer-major, expert-major within a layer. A record is one
//!   expert's six streams back to back in fill-descriptor order, so a miss
//!   is one read.
//!
//! Every offset is derived with checked arithmetic and the file length must
//! match the header exactly. A short read, a digest mismatch or a foreign
//! identity is an error, never a partially filled buffer handed on.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use paddock_models::ggml_type::GgmlType;

use super::moe_cache::{ExpertLayerLayout, ExpertPlaneLayout};
use crate::kv_tier::io::{pread_exact, pwrite_all};

const MAGIC: [u8; 4] = *b"PDXA";
const VERSION: u32 = 1;
/// magic, version, identity, n_layers, n_expert.
const FIXED_HEADER: u64 = 4 + 4 + 32 + 4 + 4;
/// ty + three dims + data and scale bytes per expert, per plane.
const PLANE_HEADER: u64 = 4 + 3 * 8 + 8 + 8;
const DIGEST: u64 = 32;
/// Sanity ceilings for a header read from disk, far above any real model;
/// they keep a corrupt header from sizing a huge allocation.
const MAX_LAYERS: u32 = 4096;
const MAX_EXPERTS: u32 = 1 << 16;

pub(super) type Identity = [u8; 32];

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn overflow() -> io::Error {
    invalid("expert artifact: offset arithmetic overflows".into())
}

fn to_u64(v: usize) -> io::Result<u64> {
    u64::try_from(v).map_err(|_| overflow())
}

fn to_usize(v: u64) -> io::Result<usize> {
    usize::try_from(v).map_err(|_| overflow())
}

/// Where each layer's records sit, derived from the layouts alone.
struct Extents {
    /// `(first record offset, record bytes)` per layer.
    layers: Vec<(u64, u64)>,
    digests_at: u64,
    len: u64,
}

impl Extents {
    fn new(layouts: &[ExpertLayerLayout], n_expert: usize) -> io::Result<Self> {
        let n_layers = to_u64(layouts.len())?;
        let n_expert = to_u64(n_expert)?;
        let digests_at = n_layers
            .checked_mul(3 * PLANE_HEADER)
            .and_then(|v| v.checked_add(FIXED_HEADER))
            .ok_or_else(overflow)?;
        let mut at = n_layers
            .checked_mul(n_expert)
            .and_then(|v| v.checked_mul(DIGEST))
            .and_then(|v| v.checked_add(digests_at))
            .ok_or_else(overflow)?;
        let mut layers = Vec::with_capacity(layouts.len());
        for l in layouts {
            let mut record = 0u64;
            for b in l.stream_bytes() {
                record = record.checked_add(to_u64(b)?).ok_or_else(overflow)?;
            }
            layers.push((at, record));
            at = record
                .checked_mul(n_expert)
                .and_then(|v| v.checked_add(at))
                .ok_or_else(overflow)?;
        }
        Ok(Self {
            layers,
            digests_at,
            len: at,
        })
    }
}

fn encode_header(identity: &Identity, layouts: &[ExpertLayerLayout]) -> io::Result<Vec<u8>> {
    let n_layers = u32::try_from(layouts.len()).map_err(|_| overflow())?;
    let n_expert = u32::try_from(layouts[0].n_expert()).map_err(|_| overflow())?;
    let mut h = Vec::new();
    h.extend_from_slice(&MAGIC);
    h.extend_from_slice(&VERSION.to_le_bytes());
    h.extend_from_slice(identity);
    h.extend_from_slice(&n_layers.to_le_bytes());
    h.extend_from_slice(&n_expert.to_le_bytes());
    for l in layouts {
        for p in &l.planes {
            h.extend_from_slice(&p.ty.raw().to_le_bytes());
            for d in &p.dims {
                h.extend_from_slice(&to_u64(*d)?.to_le_bytes());
            }
            h.extend_from_slice(&to_u64(p.data_per_expert)?.to_le_bytes());
            h.extend_from_slice(&to_u64(p.scales_per_expert)?.to_le_bytes());
        }
    }
    Ok(h)
}

/// Little-endian cursor over a header read from disk.
struct Fields<'a>(&'a [u8]);

impl Fields<'_> {
    fn take<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let (head, rest) = self
            .0
            .split_first_chunk::<N>()
            .ok_or_else(|| invalid("expert artifact: header truncated".into()))?;
        self.0 = rest;
        Ok(*head)
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }

    fn usize(&mut self) -> io::Result<usize> {
        to_usize(u64::from_le_bytes(self.take()?))
    }
}

/// Writes an artifact one expert record at a time, so producing it never
/// holds more than one record in memory. The file appears under its final
/// name only after every record and digest is written and synced; an
/// unfinished writer removes its temporary file.
pub(super) struct ExpertArtifactWriter {
    file: File,
    tmp: PathBuf,
    path: PathBuf,
    extents: Extents,
    n_expert: usize,
    digests: Vec<u8>,
    /// The next record, as `(layer, expert)`.
    next: (usize, usize),
    done: bool,
}

impl ExpertArtifactWriter {
    /// Start `path`. The temporary sibling is created exclusively: a second
    /// writer for the same artifact fails instead of interleaving.
    pub(super) fn create(
        path: &Path,
        identity: &Identity,
        layouts: &[ExpertLayerLayout],
    ) -> io::Result<Self> {
        let n_expert = layouts
            .first()
            .ok_or_else(|| invalid("expert artifact: no layers".into()))?
            .n_expert();
        if layouts.iter().any(|l| l.n_expert() != n_expert) {
            return Err(invalid(
                "expert artifact: layers disagree on the expert count".into(),
            ));
        }
        let extents = Extents::new(layouts, n_expert)?;
        let header = encode_header(identity, layouts)?;
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".partial");
        let tmp = PathBuf::from(tmp);
        let file = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        let w = Self {
            file,
            tmp,
            path: path.to_path_buf(),
            extents,
            n_expert,
            digests: Vec::with_capacity(layouts.len() * n_expert * DIGEST as usize),
            next: (0, 0),
            done: false,
        };
        pwrite_all(&w.file, &header, 0)?;
        Ok(w)
    }

    /// Append the next record in `(layer, expert)` order.
    pub(super) fn push_expert(&mut self, record: &[u8]) -> io::Result<()> {
        let (layer, expert) = self.next;
        let &(first, len) =
            self.extents.layers.get(layer).ok_or_else(|| {
                invalid("expert artifact: more records than the layout holds".into())
            })?;
        if to_u64(record.len())? != len {
            return Err(invalid(format!(
                "expert artifact: layer {layer} expert {expert} record is {} bytes, layout says {len}",
                record.len()
            )));
        }
        let off = len
            .checked_mul(to_u64(expert)?)
            .and_then(|v| v.checked_add(first))
            .ok_or_else(overflow)?;
        pwrite_all(&self.file, record, off)?;
        self.digests
            .extend_from_slice(blake3::hash(record).as_bytes());
        self.next = if expert + 1 == self.n_expert {
            (layer + 1, 0)
        } else {
            (layer, expert + 1)
        };
        Ok(())
    }

    /// Write the digests, sync, and publish under the final name.
    pub(super) fn finish(mut self) -> io::Result<()> {
        if self.next != (self.extents.layers.len(), 0) {
            return Err(invalid(format!(
                "expert artifact: finished at layer {} expert {}, before the last record",
                self.next.0, self.next.1
            )));
        }
        pwrite_all(&self.file, &self.digests, self.extents.digests_at)?;
        self.file.sync_all()?;
        std::fs::rename(&self.tmp, &self.path)?;
        self.done = true;
        Ok(())
    }
}

impl Drop for ExpertArtifactWriter {
    fn drop(&mut self) {
        if !self.done {
            let _ = std::fs::remove_file(&self.tmp);
        }
    }
}

/// An opened, header-verified artifact. Reads are positioned and take
/// `&self`, so one handle serves every layer.
pub(super) struct DiskExpertStore {
    file: File,
    layouts: Vec<ExpertLayerLayout>,
    extents: Extents,
    n_expert: usize,
    digests: Vec<u8>,
}

impl DiskExpertStore {
    /// Open `path` and check it was produced from `identity`, that its header
    /// is self-consistent and that the file is exactly as long as it says.
    pub(super) fn open(path: &Path, identity: &Identity) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut fixed = [0u8; FIXED_HEADER as usize];
        pread_exact(&file, &mut fixed, 0)?;
        let mut f = Fields(&fixed);
        if f.take::<4>()? != MAGIC {
            return Err(invalid("expert artifact: not an expert artifact".into()));
        }
        let version = f.u32()?;
        if version != VERSION {
            return Err(invalid(format!(
                "expert artifact: format version {version}, this build reads {VERSION}"
            )));
        }
        if &f.take::<32>()? != identity {
            return Err(invalid(
                "expert artifact: produced from a different model or repack; regenerate it".into(),
            ));
        }
        let n_layers = f.u32()?;
        let n_expert = f.u32()?;
        if n_layers == 0 || n_layers > MAX_LAYERS || n_expert == 0 || n_expert > MAX_EXPERTS {
            return Err(invalid(format!(
                "expert artifact: implausible header ({n_layers} layers, {n_expert} experts)"
            )));
        }
        let (n_layers, n_expert) = (n_layers as usize, n_expert as usize);
        let mut planes = vec![0u8; n_layers * 3 * PLANE_HEADER as usize];
        pread_exact(&file, &mut planes, FIXED_HEADER)?;
        let mut f = Fields(&planes);
        let mut layouts = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let mut plane = || -> io::Result<ExpertPlaneLayout> {
                let ty = GgmlType::from_raw(f.u32()?);
                let dims = [f.usize()?, f.usize()?, f.usize()?];
                let data = f.usize()?;
                let scales = f.usize()?;
                let total = |per: usize| per.checked_mul(dims[2]).ok_or_else(overflow);
                ExpertPlaneLayout::new(&dims, ty, total(data)?, total(scales)?)
                    .map_err(|e| invalid(format!("expert artifact: {e}")))
            };
            let l = ExpertLayerLayout::new([plane()?, plane()?, plane()?])
                .map_err(|e| invalid(format!("expert artifact: {e}")))?;
            if l.n_expert() != n_expert {
                return Err(invalid(format!(
                    "expert artifact: a layer holds {} experts, header says {n_expert}",
                    l.n_expert()
                )));
            }
            layouts.push(l);
        }
        let extents = Extents::new(&layouts, n_expert)?;
        if file_len != extents.len {
            return Err(invalid(format!(
                "expert artifact: {file_len} bytes on disk, header describes {}",
                extents.len
            )));
        }
        let mut digests = vec![0u8; n_layers * n_expert * DIGEST as usize];
        pread_exact(&file, &mut digests, extents.digests_at)?;
        Ok(Self {
            file,
            layouts,
            extents,
            n_expert,
            digests,
        })
    }

    pub(super) fn n_layers(&self) -> usize {
        self.layouts.len()
    }

    pub(super) fn layout(&self, layer: usize) -> Option<&ExpertLayerLayout> {
        self.layouts.get(layer)
    }

    /// Bytes of one record (one expert's six streams) in `layer`.
    pub(super) fn record_bytes(&self, layer: usize) -> Option<usize> {
        self.extents
            .layers
            .get(layer)
            .and_then(|&(_, len)| usize::try_from(len).ok())
    }

    /// Read `expert` of `layer` into `dst` (exactly one record long) and
    /// verify it against its digest.
    pub(super) fn read_expert(
        &self,
        layer: usize,
        expert: usize,
        dst: &mut [u8],
    ) -> io::Result<()> {
        let &(first, len) = self
            .extents
            .layers
            .get(layer)
            .ok_or_else(|| invalid(format!("expert artifact: no layer {layer}")))?;
        if expert >= self.n_expert {
            return Err(invalid(format!(
                "expert artifact: expert {expert} over {}",
                self.n_expert
            )));
        }
        if to_u64(dst.len())? != len {
            return Err(invalid(format!(
                "expert artifact: {} byte buffer for a {len} byte record",
                dst.len()
            )));
        }
        let off = len
            .checked_mul(to_u64(expert)?)
            .and_then(|v| v.checked_add(first))
            .ok_or_else(overflow)?;
        pread_exact(&self.file, dst, off)?;
        let at = (layer * self.n_expert + expert) * DIGEST as usize;
        if blake3::hash(dst).as_bytes()[..] != self.digests[at..at + DIGEST as usize] {
            return Err(invalid(format!(
                "expert artifact: layer {layer} expert {expert} fails its digest"
            )));
        }
        Ok(())
    }
}

/// Proof of holding one staging buffer. Not `Clone`: a buffer is released
/// exactly once, by giving the lease back.
#[derive(Debug)]
pub(super) struct StagingLease(usize);

/// A fixed set of equally sized staging buffers, allocated up front. The pool
/// never grows: when every buffer is leased, `acquire` returns `None` and the
/// caller waits for an in-flight copy to complete before releasing one.
pub(super) struct StagingPool<B> {
    bufs: Vec<B>,
    free: Vec<usize>,
}

impl<B: AsRef<[u8]> + AsMut<[u8]>> StagingPool<B> {
    /// `count` buffers of `len` bytes each from `alloc`.
    pub(super) fn new(
        count: usize,
        len: usize,
        mut alloc: impl FnMut(usize) -> io::Result<B>,
    ) -> io::Result<Self> {
        if count == 0 || len == 0 {
            return Err(invalid(format!(
                "expert staging: {count} buffers of {len} bytes"
            )));
        }
        let mut bufs = Vec::with_capacity(count);
        for _ in 0..count {
            let b = alloc(len)?;
            if b.as_ref().len() != len {
                return Err(invalid(format!(
                    "expert staging: allocator returned {} bytes, asked {len}",
                    b.as_ref().len()
                )));
            }
            bufs.push(b);
        }
        Ok(Self {
            bufs,
            free: (0..count).rev().collect(),
        })
    }

    pub(super) fn capacity(&self) -> usize {
        self.bufs.len()
    }

    pub(super) fn available(&self) -> usize {
        self.free.len()
    }

    pub(super) fn acquire(&mut self) -> Option<StagingLease> {
        self.free.pop().map(StagingLease)
    }

    pub(super) fn buf(&self, lease: &StagingLease) -> &[u8] {
        self.bufs[lease.0].as_ref()
    }

    pub(super) fn buf_mut(&mut self, lease: &StagingLease) -> &mut [u8] {
        self.bufs[lease.0].as_mut()
    }

    pub(super) fn release(&mut self, lease: StagingLease) {
        self.free.push(lease.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: Identity = [7u8; 32];
    const N_EXPERT: usize = 4;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pdk-xa-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("temp dir");
        d
    }

    fn plane(rows: usize, data: usize, scales: usize) -> ExpertPlaneLayout {
        ExpertPlaneLayout::new(
            &[256, rows, N_EXPERT],
            GgmlType::Q4K,
            data * N_EXPERT,
            scales * N_EXPERT,
        )
        .expect("valid plane")
    }

    /// Two layers with different record sizes, so a per-layer offset bug
    /// shows up as a wrong record rather than a lucky one.
    fn layouts() -> Vec<ExpertLayerLayout> {
        vec![
            ExpertLayerLayout::new([plane(8, 64, 12), plane(8, 64, 12), plane(4, 32, 6)])
                .expect("layer 0"),
            ExpertLayerLayout::new([plane(16, 128, 24), plane(16, 128, 24), plane(8, 64, 12)])
                .expect("layer 1"),
        ]
    }

    fn record(layouts: &[ExpertLayerLayout], layer: usize, expert: usize) -> Vec<u8> {
        let len: usize = layouts[layer].stream_bytes().iter().sum();
        (0..len)
            .map(|i| (i as u8) ^ ((layer as u8) << 6) ^ ((expert as u8) << 3))
            .collect()
    }

    fn write(path: &Path, layouts: &[ExpertLayerLayout]) {
        let mut w = ExpertArtifactWriter::create(path, &ID, layouts).expect("create");
        for l in 0..layouts.len() {
            for e in 0..N_EXPERT {
                w.push_expert(&record(layouts, l, e)).expect("push");
            }
        }
        w.finish().expect("finish");
    }

    fn kind(r: io::Result<impl Sized>) -> io::ErrorKind {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.kind(),
        }
    }

    #[test]
    fn round_trips_every_record_and_layout() {
        let path = dir("round").join("experts.pdxa");
        let ls = layouts();
        write(&path, &ls);
        assert!(!path.with_extension("pdxa.partial").exists());
        let s = DiskExpertStore::open(&path, &ID).expect("open");
        assert_eq!(s.n_layers(), 2);
        for (l, want) in ls.iter().enumerate() {
            assert_eq!(s.layout(l), Some(want));
            let len = s.record_bytes(l).expect("layer");
            assert_eq!(len, want.stream_bytes().iter().sum::<usize>());
            // reverse order: reads are positioned, not sequential
            for e in (0..N_EXPERT).rev() {
                let mut buf = vec![0u8; len];
                s.read_expert(l, e, &mut buf).expect("read");
                assert_eq!(buf, record(&ls, l, e), "layer {l} expert {e}");
            }
        }
    }

    #[test]
    fn open_rejects_foreign_or_malformed_files() {
        let d = dir("open");
        let path = d.join("experts.pdxa");
        write(&path, &layouts());
        assert_eq!(
            kind(DiskExpertStore::open(&path, &[8u8; 32])),
            io::ErrorKind::InvalidData
        );

        let good = std::fs::read(&path).expect("read back");
        let probe = |bytes: &[u8]| {
            let p = d.join("probe.pdxa");
            std::fs::write(&p, bytes).expect("write probe");
            kind(DiskExpertStore::open(&p, &ID))
        };
        let mut b = good.clone();
        b[0] = b'X';
        assert_eq!(probe(&b), io::ErrorKind::InvalidData, "magic");
        let mut b = good.clone();
        b[4] = 2;
        assert_eq!(probe(&b), io::ErrorKind::InvalidData, "version");
        assert_eq!(
            probe(&good[..good.len() - 1]),
            io::ErrorKind::InvalidData,
            "short"
        );
        let mut b = good.clone();
        b.push(0);
        assert_eq!(probe(&b), io::ErrorKind::InvalidData, "trailing bytes");
        // first plane's expert count (dims[2]) no longer matches the header
        let mut b = good.clone();
        let dims2 = (FIXED_HEADER + 4 + 2 * 8) as usize;
        b[dims2] = 5;
        assert_eq!(probe(&b), io::ErrorKind::InvalidData, "plane geometry");
        assert_eq!(
            probe(&good[..10]),
            io::ErrorKind::UnexpectedEof,
            "truncated header"
        );
    }

    #[test]
    fn read_rejects_corruption_and_bad_requests() {
        let path = dir("read").join("experts.pdxa");
        let ls = layouts();
        write(&path, &ls);
        let len0 = ls[0].stream_bytes().iter().sum::<usize>() as u64;
        let first = Extents::new(&ls, N_EXPERT).expect("extents").layers[0].0;
        // flip one byte inside layer 0 expert 2
        let mut bytes = std::fs::read(&path).expect("read back");
        bytes[(first + 2 * len0 + 5) as usize] ^= 1;
        std::fs::write(&path, &bytes).expect("rewrite");

        let s = DiskExpertStore::open(&path, &ID).expect("length is unchanged");
        let mut buf = vec![0u8; len0 as usize];
        assert_eq!(
            kind(s.read_expert(0, 2, &mut buf)),
            io::ErrorKind::InvalidData
        );
        s.read_expert(0, 1, &mut buf)
            .expect("neighbour still reads");
        assert_eq!(buf, record(&ls, 0, 1));
        assert_eq!(
            kind(s.read_expert(0, N_EXPERT, &mut buf)),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            kind(s.read_expert(2, 0, &mut buf)),
            io::ErrorKind::InvalidData
        );
        let mut short = vec![0u8; len0 as usize - 1];
        assert_eq!(
            kind(s.read_expert(0, 0, &mut short)),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn writer_is_exclusive_ordered_and_leaves_nothing_on_failure() {
        let d = dir("writer");
        let path = d.join("experts.pdxa");
        let partial = d.join("experts.pdxa.partial");
        let ls = layouts();

        let mut w = ExpertArtifactWriter::create(&path, &ID, &ls).expect("create");
        assert_eq!(
            kind(ExpertArtifactWriter::create(&path, &ID, &ls)),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(kind(w.push_expert(&[0u8; 3])), io::ErrorKind::InvalidData);
        w.push_expert(&record(&ls, 0, 0)).expect("push");
        assert_eq!(kind(w.finish()), io::ErrorKind::InvalidData);
        assert!(!partial.exists() && !path.exists());

        let mut w = ExpertArtifactWriter::create(&path, &ID, &ls).expect("create again");
        for l in 0..ls.len() {
            for e in 0..N_EXPERT {
                w.push_expert(&record(&ls, l, e)).expect("push");
            }
        }
        assert_eq!(
            kind(w.push_expert(&record(&ls, 0, 0))),
            io::ErrorKind::InvalidData
        );
        w.finish().expect("finish");
        assert!(path.exists() && !partial.exists());

        let mixed = [
            ls[0].clone(),
            ExpertLayerLayout::new([
                ExpertPlaneLayout::new(&[256, 8, 2], GgmlType::Q4K, 128, 24).expect("plane"),
                ExpertPlaneLayout::new(&[256, 8, 2], GgmlType::Q4K, 128, 24).expect("plane"),
                ExpertPlaneLayout::new(&[256, 4, 2], GgmlType::Q4K, 64, 12).expect("plane"),
            ])
            .expect("layer"),
        ];
        assert_eq!(
            kind(ExpertArtifactWriter::create(
                &d.join("mixed.pdxa"),
                &ID,
                &mixed
            )),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            kind(ExpertArtifactWriter::create(
                &d.join("empty.pdxa"),
                &ID,
                &[]
            )),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn staging_pool_is_bounded() {
        let mut pool = StagingPool::new(2, 16, |len| Ok(vec![0u8; len])).expect("pool");
        assert_eq!((pool.capacity(), pool.available()), (2, 2));
        let a = pool.acquire().expect("first");
        let b = pool.acquire().expect("second");
        assert!(pool.acquire().is_none());
        pool.buf_mut(&a).fill(1);
        pool.buf_mut(&b).fill(2);
        assert!(pool.buf(&a).iter().all(|&v| v == 1));
        assert!(pool.buf(&b).iter().all(|&v| v == 2));
        pool.release(a);
        assert_eq!(pool.available(), 1);
        let c = pool.acquire().expect("released buffer is reused");
        assert!(pool.buf(&c).iter().all(|&v| v == 1));

        assert!(StagingPool::new(0, 16, |len| Ok(vec![0u8; len])).is_err());
        assert!(StagingPool::new(2, 0, |len| Ok(vec![0u8; len])).is_err());
        assert!(StagingPool::new(2, 16, |len| Ok(vec![0u8; len - 1])).is_err());
        let mut calls = 0;
        let failing = StagingPool::<Vec<u8>>::new(3, 16, |_| {
            calls += 1;
            Err(io::Error::other("refused"))
        });
        assert!(failing.is_err());
        assert_eq!(calls, 1, "allocation stops at the first refusal");
    }

    #[test]
    fn artifact_layout_is_compared_by_value() {
        // The cache is built from the pinned planes' layout; a disk artifact
        // serves that cache only if its layout is the same value.
        let path = dir("same").join("experts.pdxa");
        let ls = layouts();
        write(&path, &ls);
        let s = DiskExpertStore::open(&path, &ID).expect("open");
        assert_eq!(s.layout(1), Some(&ls[1]));
        assert_ne!(s.layout(0), Some(&ls[1]));
    }
}
