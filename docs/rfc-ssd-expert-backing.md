# RFC: bounded disk-backed MoE expert cache (draft)

## Problem

The existing CUDA MoE offload stores the complete repacked gate/up/down planes in device-mapped, page-locked host memory. This is a good fast path but cannot serve a model whose expert planes exceed the host's lockable RAM. KV tiering is not expert-weight tiering. The target is a consumer box (32 GiB RAM, an 8 GiB sm_86 card), starting with a supported 35B MoE and attempting a supported low-bit 180B Flash Next pack only after a measured fit check.

Related upstream: [#8](https://github.com/truespar/paddock/issues/8), [#16](https://github.com/truespar/paddock/issues/16), [#20](https://github.com/truespar/paddock/issues/20). This is a proposed extension, not a bug in the current design.

## Design constraints

- Existing resident and pinned modes remain the default, with identical CUDA graph execution, kernel ABI and cache layout. No host synchronization or disk dispatch in the existing hot path.
- Reuse the existing repacked KQ/IQ format, expert IDs, slot planes, router, quantized compute and KV scheduler. No second inference engine; no upstream code copied from other engines.
- Storage and miss scheduling are separate concerns. Disk is not a device pointer and cannot satisfy HostMappedKq's Deref contract.
- Start with CUDA Qwen35 (Qwen3.6-35B-A3B), then Qwen4Exp; one request, no speculation/overlap, bounded prompt waves. Explicitly reject unsupported quantization, insufficient slots or unsupported execution modes. Never silently degrade the pinned mode or advertise unsupported capabilities.

## Decisions

- 2026-11-26: wire qwen35 (Qwen3.6-35B-A3B) before qwen4exp. Parity needs a model that fits both pinned and disk backing on the 32 GiB target host; the 180B qwen4exp model fits only on disk.

## Patch stack (independent of the fork's Nix packaging)

1. **Extract layout** (included in this draft): separate slot geometry from host source pointers in gpu/moe_cache.rs. Existing new_expert_cache signature and fill ABI are unchanged. Tests are provided but still need execution.
2. **Storage source**: introduce a private ExpertStore trait (or a small enum for two implementations) after the real second implementation exists. Keep a pinned source adapter owning HostMappedKq as today. Disk source owns an immutable verified repacked-weight artifact, checked offsets/lengths and a bounded pool of pinned staging buffers. Prefer a single enum dispatch at model load or layer boundary, not a trait call for each weight. No new exported stable API or plugin system.
3. **Host-assisted cache fills**: extract a separate fill policy. Keep current device-resolve/device-fill path as is. On the disk path, resolve routed IDs, copy bounded miss jobs/count to host outside graph capture, perform checked positioned reads into staging, enqueue H2D copies into existing slot planes, wait on completion in stream order, then launch unchanged expert compute. Existing resolve mutates LRU state before fill; on failure invalidate affected cache/state and fail the request or retire model. Never permit a stale/partially-filled slot to be consumed. Account for all six data/scale streams; maintain staging buffer ownership through CUDA completion.
4. **Wire one family**: qwen35 first (gpu_model/qwen35 enable_moe_cache), qwen4exp (load_gguf.rs kq_seat, forward.rs enable_moe_cache) after; use explicit source/capability selection. Update shared KqSeat only if required; all matches must handle or reject disk mode, not silently dereference a disk seat. Disk mode must require a usable slot cache (no existing zero-copy fallback). Review decode_step, decode_step_batch, prefill/wave and speculative graph callers. Disable graph capture for this mode at model setup, with a precise startup reason, until a segmented replay strategy is proven. Do not synchronize or perform filesystem I/O in capture or CUDA host callbacks. No speculative router lookahead in initial implementation.
5. **Configuration and estimator**: opt-in disk backing in the existing [moe_offload] block, default pinned. Explicit host staging/cache byte budget, optional artifact directory if required. Keep vram_gb/ctx/batch semantics. Surface required dense VRAM, disk size, repack transient peak, staging, RAM and slot geometry. Prefer platform-appropriate positional file APIs and portable tests; no Linux-specific API in common storage contract. Reuse existing metrics and engine error conventions.

Illustrative internal seam, not a final API:
```rust
trait ExpertStore {
    fn layout(&self) -> &ExpertLayerLayout;
    fn backing_kind(&self) -> BackingKind;
}

trait HostMissReader {
    fn read_expert(&self, id: ExpertId, dst: &mut ExpertStaging)
        -> Result<(), GpuError>;
}
// Pinned backing keeps its existing device-address fill.
// Disk backing implements HostMissReader. Only disk uses host-assisted fills.
// Dispatch once outside graph capture. The owner retains all backing memory.
```

Do not force pinned backing into a host-read abstraction just to share code. The first refactor is intentionally layout-only; the capability/policy split is the next review point. Do not introduce a dummy SSD backend that returns null device addresses.

## Artifact format and bounded loading

The first experiment can use a generated repacked artifact because the current GPU repack path consumes GGUF and produces separate data/scale streams. Verify per-expert contiguity and identical bytes for each supported quant; use actual layout rather than deriving offsets from model name. Repack in bounded per-expert/chunk units if full plane temporary allocation does not fit. Never allocate the entire repacked model in host RAM before writing the artifact.

Version manifest with model identity/hash, quantization, repack ABI/geometry, tensor names, expert count, exact offsets/lengths and content checksums. Use checked arithmetic, exact reads (short read/error is fatal), atomic temp-to-final publication, exclusive writer, no unverified file reuse. Record disk budget including original GGUF + artifact + temporary overhead. Prefer simple buffered positioned reads first; compare direct I/O only when measured.

## Verification and upstream gates

- Offline: layout/stride validation, checked offset arithmetic, fixture reads, short read/corruption and wrong model identity, budget limits, no unbounded staging or file writes, cache miss state failure and retry safety.
- GPU: byte equality of all six repacked streams; greedy/logit parity for identical weights and forced token sequences against current pinned path, including cache hit, miss, eviction, duplicate routed IDs, prefill waves and repeated decode. Use tolerances consistent with existing upstream parity tests; don't silently relax. Fault injection for I/O failure before/during fill, no stale cache output.
- Existing mode: run upstream relevant engine tests and compare graph-on baseline, same model/fixture, hit rates and decode/TTFT. No regression attributable to the seam; publish measured spread/sample counts, not an arbitrary speed claim.
- Disk mode: first 35B-A3B with deliberately constrained RAM; then supported 180B if estimator passes. Real SSD reads (not just file mapping), 4K context, one request, 5 cold/20 distinct warm and 30-minute stability, bounded RAM/VRAM, cache hit/miss and I/O counters, no OOM or GPU reset. Record PCIe link under load. Measure decode, prefill, TTFT and I/O stalls; compare same weights pinned when host can fit. No new defaults or performance promises.

## Fork maintenance

Keep a dedicated feature branch based on upstream main, separate from packaging work. Rebase the small commits in order, and keep the baseline upstream test matrix plus disk-mode fixtures. Avoid formatting churn, dependency upgrades, model-specific duplicated loops, and unrelated manager/UI changes. If upstream wants a different abstraction, adapt the private seam; if rejected, carry the isolated patch stack in the fork and pin the verified commit downstream. Do not submit upstream until compilation, parity, measurements and independent review pass.

## Implementation status

Steps 1 to 4 exist for qwen35. On an 8 GiB sm_86 card with the 35B-A3B UD file and 32 slots, the pinned path after the seam matched resident bit for bit and main's tokens and decode rate; disk-mode parity and measurements have not run yet.

- `gpu/expert_store.rs`: the artifact (sealed header, per-record BLAKE3, exact positioned reads, exclusive temp-then-rename writer) and the fixed staging pool. CPU tests.
- `gpu/moe_disk.rs`: the host miss path (the resolve kernel's LRU walk on the host, token ranges whose experts fit the slots, invalidation of every slot a failed fill assigned, a broken state after a failed wait) behind a two-call device seam; CPU tests drive it with a fake device that applies copies at wait time. The CUDA side (`CacheCopier`, `PinnedBuf`, routing upload, row views) is compile-checked only.
- qwen35: `PADDOCK_MOE_DISK=<artifact>` seats layout-only experts, requires a slot cache of at least top-k slots, runs prefill and decode eagerly, launches each token range over its own rows, and refuses CUDA graph capture, batched serving (a startup config error), DFlash and MTP speculation. qwen4exp refuses the switch.
- The artifact is written straight from the GGUF, without loading the model: one layer's experts are repacked in VRAM at a time and copied down one expert record at a time, so host memory holds one record.
- `tests/gpu_qwen36_moe_disk.rs`: writes the artifact, then a pinned reference, then checks bit-exact parity (prefill logits and greedy tokens, token-batched class on both sides) and reports TTFT, decode, misses, read rate, staging waits, device reads and peak RSS, plus the same runs pinned.

Known gaps, in order of what blocks the 180B target:

- The writer needs one layer's repacked experts (plus its raw upload) to fit in VRAM.
- Only the single-sequence eager path serves; the runner's batched lane refuses at startup.
- Prefill syncs the host once per token range and layer; reads are buffered and synchronous (no prefetch, no direct I/O) until measured.
- The identity hashes the kernel pack version and each expert tensor's name, type, shape, length and first and last 64 KiB, not every byte.
- No parent directory fsync after publishing the artifact.

