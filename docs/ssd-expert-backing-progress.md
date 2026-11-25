# SSD expert backing: progress on asus-b460m

Host: i9-10900K, 32 GiB RAM, RTX 3070 8 GiB (46 SMs, sm_86, PCIe 3.0 x16), Kingston SA2000 NVMe.
Model: Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf. Pack: paddock-cuda-sm86-0.1.21 (nix store). Branch: asus/ssd-expert-backing.
Run helper: ~/paddock-runs/w/run.sh (systemd-run MemoryMax cap, swap off, nvidia-smi every 200 ms); logs in ~/paddock-runs/w/logs.

## Proven, with numbers

### R6 failure: root cause (neither host nor GPU memory exhaustion)

`prefill: Gpu(Launch(2))` is the pack's K-split f32 router GEMM refusing inside a CUDA graph capture.

- Not host memory: it reproduces with the artifact write skipped (R6b, artifact kept, 28G cap).
- Not GPU memory: GPU peak 5809 MiB of 8192 (R6b, sampled every second); 5851 MiB on the passing rerun.
- `PADDOCK_PANIC_LAUNCH=1` puts the failing launch at `batch_ops.rs:612`, `matvec_f32_raw` (the `matvec_f32_batch` router/decay plane).
- On dies under 128 SMs `pd_matvec_f32_batch` takes the K-split arm from 256 rows (`ks_small`). That arm gets its partials from `pd_part_scratch`, which never allocates during a stream capture, and `pd_f32nt_ks_go` then returns `cudaErrorMemoryAllocation` (2) when the stream has no big enough plane (`packs/cuda/src/gemm/f32_qkv.cuh:584`, `part_scratch.cuh`).
- The pinned `prefill()` captures a graph the first time it sees a length. The same 600-token pinned prefill passes with `PADDOCK_NO_PREFILL_GRAPH=1` (R8-600-eager) or `PADDOCK_NO_F32NT_KS=1` (R6c), and fails without them (R8-600, 17.7 s).
- This is in main's pinned path, not in the seam. The 4000-token pinned prefill (R8) captures without failing; why a plane exists there and not at 600 is not established yet.
- Fix (test only, pinned fast path untouched): the reference prefill runs eager like the disk path (`PADDOCK_NO_PREFILL_GRAPH=1` in `moe_disk_write_artifact`). Commit 3b584da. The pack bug is an upstream report draft, see open questions.

### R6 and R7 now run

- R6d (28G cap, artifact kept): passes in 44 s, GPU peak 5845 MiB, reference written.
- New GGUF writer (merged from pool head 0d0be46): artifact written in 67.1 s under an 8G cap and byte-equal to the pinned-load artifact (`cmp` over 20669863152 bytes). VmHWM 12.4 GiB counts mmapped GGUF pages.
- R7a (8G cap, swap off, 32 slots, warm=1): parity passes. `parity: 0 of 248320 prefill logits differ bitwise, tokens equal=true` (600-token prompt, 32 greedy tokens). Batched serving refuses with the startup sentence.

| R7a, 32 slots, 8G cap | value |
|---|---|
| cold TTFT, 4000 tokens | 485.7 s (8.2 tok/s), 365121 misses of 1280000 rows (hit 71.5%), read 704031 MiB at 1521 MiB/s, staging waits 34773 |
| cold decode | 6.73 tok/s over 64 tokens, hit 73.1%, read 10572 MiB at 1309 MiB/s |
| warm decode | 5.63 tok/s, hit 73.1%, 1063 MiB/s |
| peak RSS (VmHWM) | 3.0 GiB; cgroup memory peak 8G (page cache included) |
| GPU peak | 7303 MiB |

Pinned on the same harness (R8 this morning, graphs on, 28G cap): TTFT 149.1 s for 4000 tokens (26.8 tok/s), decode 34.3 tok/s.

Reading: token-range prefill thrashes the 32 slots (each range holds at most 32 distinct experts, so a 4000-token prompt reads the 19 GiB artifact about 36 times). Decode is read bound: about 165 MiB a token at about 1.3 GiB/s.

### Environment

- Dev shell: `asus/devshell-driver-lib` (one commit, 355df36, on top of PR 1's codex/hpc-research) adds a shellHook that puts /run/opengl-driver/lib on LD_LIBRARY_PATH. Checked: `env -u LD_LIBRARY_PATH nix develop . --command` prints `LD=/run/opengl-driver/lib`; nixfmt and statix pass. To be folded into PR 1.
- Disk gates fail loudly: with PADDOCK_HEAVY_TESTS set, `gpu_qwen36_moe_disk` sets PADDOCK_STRICT_GATES, so a missing driver, pack or model fails the test; a missing artifact variable panics (105a0ba).

## Next

1. Expert-major prefill on the disk path (read each routed expert once per layer per prompt) instead of token ranges.
2. Slot count sweep, staging pool size, read sizes; decode prefetch or overlap.
3. Rerun the R4 pinned comparison after any change near the pinned path.

## Open questions

- Upstream pack bug: K-split router arm returns error 2 under capture with no partials plane. Draft an issue for truespar/paddock (owner decision to file).
- Why the 4000-token pinned capture does not hit it.
- The parity prompt is synthetic (random ids) and the greedy output loops ([238, 197, 197, 1, 198, ...]); a real text prompt would make token parity a stronger check.
