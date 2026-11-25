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

### Expert-major prefill (waves), 44b8475 + 36c068e

Token-range chunking read the whole artifact about 36 times for a 4000-token prompt. The disk path now plans waves on the host: the launch's distinct experts in id order, at most `slots` a wave. A launch that fits one wave (decode) runs the token-batched pair exactly as before. A wider one runs upstream's LIST pair kernels (`kquant_moe_gate_up_list`, `kquant_moe_down_list`) over each wave's pairs, as qwen4exp's `kq_moe_routed_waves` does, down accumulating into a zeroed `proj`. Waves depend on the routing only, never on what the slots hold. The first version ordered resident experts first, and the same prompt then gave different logits depending on history (synthetic max |diff| 0.203 in one run, 0.189 in the next).

Parity changed class: waves sum a token's experts in wave order. Calibration on the pinned side alone (same 3 prompts): the sorted prefill against the token-batched one gives max |diff| 0.291 (text) and 0.167 (synthetic), with equal greedy tokens. Waves against token-batched: 0.215 and 0.192. So the gate is now (tests/gpu_qwen36_moe_disk.rs):

- one-token prompt (every launch is one wave): logits bit for bit, greedy tokens equal;
- text prompt (37 tokens, gpu_qwen36_moe_host's prompt plus its greedy continuation) and 600 synthetic tokens: teacher-forced on pinned's tokens, prefill mean |d|/rms < 0.10 and top-1 at least 90%, upstream's tolerance for regrouped sums (gpu_nemotron_gguf `teacher_forced`); free-running greedy equality reported.

R7f (8G cap, swap off, 32 slots), `parity_and_measure` output:

```
parity one token (1 tokens): 0 of 248320 prefill logits differ bitwise, |d|/rms 0.00000, greedy tokens equal=true, 4598 misses of 10240 rows
parity text (37 tokens): 248319 of 248320 prefill logits differ bitwise, |d|/rms 0.01099, greedy tokens equal=true, 6044 misses of 21760 rows
  teacher-forced top-1 32/32
parity synthetic (600 tokens): 248319 of 248320 prefill logits differ bitwise, |d|/rms 0.02708, greedy tokens equal=true, 8931 misses of 201920 rows
  teacher-forced top-1 32/32
```

Near-tie seen: with the slot count at 16 or 40 the wave grouping changes, and the text prompt's free-running output parts at step 2, where disk logits are 18.0756 vs 18.0383 (16 slots) and 18.0550 vs 17.9793 (40 slots). Teacher-forced top-1 stays 31/32 or better. With 32 slots and id-order waves the free-running tokens are equal.

### Measurements (4000-token prompt, 64 decode tokens, one sequence)

Cold = artifact page cache dropped with POSIX_FADV_DONTNEED first. Each row is one process: run 0 cold, run 1 warm.

| config | cold TTFT | warm TTFT | decode (cold / warm) | decode hit | prefill read | GPU peak |
|---|---|---|---|---|---|---|
| pinned (R8, 28G, graphs) | 149.1 s | 149.1 s | 33.3 / 34.3 tok/s | 72.3% overall | zero-copy | 7355 MiB |
| token ranges (R7a, 8G) | 485.7 s | - | 6.73 / 5.63 | 73.1% | 704031 MiB at 1521 MiB/s | 7303 MiB |
| waves, 1 reader, 8G (R7f) | 32.4 s | 31.2 s | 8.17 / 8.28 | 72.1% | 15538 MiB at 941 MiB/s | 7271 MiB |
| waves, 4 readers, 8G (R12-r4-st8) | 29.2 s | 28.3 s | 9.55 / 9.61 | 72.1% | 15538 MiB in 12.24 s (1269 MiB/s) | |
| waves, 4 readers, 32 staging, 8G | 28.8 s | 27.8 s | 9.51 / 9.61 | 72.1% | 1313 MiB/s, 0 staging waits | |

Host memory cap (1 reader, 32 slots): 4G 32.3 s TTFT, decode 6.49 / 6.80 tok/s at 1300 MiB/s; 8G 8.2 tok/s; 16G 11.67 / 11.92 tok/s at 2.7 GB/s (page cache serves part of decode). 27G (artifact fully cached by run 1): warm TTFT 21.5 s with reads 4.95 s at 3.1 GB/s, decode 13.3 tok/s. Peak RSS (VmHWM) 3.0 GiB in every disk run; it includes mmapped GGUF pages.

Slot count (1 reader, 8G): 16 slots TTFT 30.6 / 28.8 s, decode 5.65 / 6.13 tok/s, hit 46.7%, GPU 6055 MiB; 24 slots 31.6 / 30.0 s, 7.37 / 7.67 tok/s, hit 63.7%, 6663 MiB; 32 slots above; 40 slots ran out of VRAM in the 4000-token prefill (`prefill: Gpu(OutOfMemory)`), the scratch for 4000 rows does not leave room.

R5 baseline (fio, this morning): 2 MiB random direct reads 1325 MiB/s at QD1, 1345 MiB/s with 4 jobs. Four readers now reach that ceiling on cold prefill.

### Pinned path after these changes (R4b, 36c068e, 26G cap, 5 interleaved runs each)

- Greedy tokens byte-equal to main in all 10 runs; expert cache 19840 rows, 5845 misses (70.5%) on both.
- Decode rerun line: main 31.13, 31.12, 31.12, 31.14, 31.09 (median 31.12); branch 31.09, 31.13, 31.13, 31.13, 31.13 (median 31.13).
- R3 `kq_moe_pair_host_mapped_bitmatches_resident` on the branch: `gate_up bitwise=true down bitwise=true`.

### Review (code-reviewer subagent, evidence not approval)

No critical finding on the wave path. Fixed in 36c068e: the per-layer `max_ctx x top-k` routing scratch is now one shared `DiskRouting`; the LIST kernels are checked at seat; `DiskExperts::fill` returns an error instead of panicking when given more distinct experts than slots. Found outside this stack, upstream: qwen4exp `kq_moe_routed_waves` computes the input sums once before its wave loop, but the down sums overwrite `d_ssums` inside it, so waves after the first read down sums when both gate/up and down need them (forward.rs:5939 and 5965). The disk path recomputes them per wave. Not verified on a GPU.

## Next

1. Overlap a wave's reads with the previous wave's kernels: every fill starts with a full stream sync, and each pageable routing upload syncs the stream too. Warm-from-RAM TTFT is 21.5 s with only 5 s of reads, so most of the remaining time is serialized GPU work.
2. Full R7 at the chosen config: 5 cold + 20 warm, memory.peak under a tight cap.
3. Rerun R4 after any change near the pinned path.

## Open questions

- Upstream pack bug: K-split router arm returns error 2 under capture with no partials plane. Draft an issue for truespar/paddock (owner decision to file).
- Why the 4000-token pinned capture does not hit it.
- Bounded RAM criterion: cgroup memory.peak counts page cache, so with buffered reads it reaches any cap; 4G works (RSS 3.0 GiB). Direct I/O would make the bound exact but loses the page-cache tier that lifts decode from 6.5 (4G) to 11.9 tok/s (16G).
- The parity prompt is synthetic (random ids) and the greedy output loops ([238, 197, 197, 1, 198, ...]); a real text prompt would make token parity a stronger check.
