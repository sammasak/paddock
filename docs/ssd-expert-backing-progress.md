# SSD expert backing: progress on asus-b460m

GPU NEEDED (2026-11-25 ~20:55, branch asus/biggest-moe): stop paddock.service, then I run, one at a time: (1) the Qwen3.5-122B-A10B UD-Q2_K_XL disk-backed serve (`~/paddock-runs/w/serve-122/start.sh`, user unit `paddock-big`, port 11541; the first start writes its ~37 GB artifact), one chat completion and one tool call; (2) the dense_host bit-exactness gate on the 35B; (3) later the 397B once its download (paused behind the 122B, now resumed) finishes. GGUF of the 122B is complete and checksum-verified (sha256 be44a62a...).

Host: i9-10900K, 32 GiB RAM, RTX 3070 8 GiB (46 SMs, sm_86, PCIe 3.0 x16), Kingston SA2000 NVMe.
Model: Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf. Pack: paddock-cuda-sm86-0.1.21 (nix store). Branch: asus/ssd-expert-backing.
Run helper: ~/paddock-runs/w/run.sh (systemd-run MemoryMax cap, swap off, nvidia-smi every 200 ms); logs in ~/paddock-runs/w/logs.

## Biggest MoE that runs on asus (owner goal, 2026-11-25 evening; branch asus/biggest-moe)

Fable consult (read-only, repo + web) ranked ports: Qwen3.8-Flash-Next via a qwen4exp disk port (66-111 GB, 20-30 h, 1.3-5 tok/s), Laguna S 2.1 118B-A8B / Kolibri 1 78B-A3.5B via a laguna disk port (12-20 h; Kolibri 5-9 tok/s estimated), gpt-oss-120b and Nemotron Super out (MXFP4 slot ABI, 22 active experts). It missed that the qwen35moe family itself has bigger releases, which the disk path serves with no port:

| model, quant (unsloth) | GGUF | dense (non-expert) | experts | layout | blocker |
|---|---|---|---|---|---|
| Qwen3.5-397B-A17B UD-IQ1_M | 106.8 GB (4 shards) | 7.15 GB | 99.7 GB, ~3.2 MB a record | 60 x 512 x top-10, expert ff 1024, embd 4096 | dense does not fit 8 GB beside slots/KV/scratch: needs dense weights served from host |
| Qwen3.5-122B-A10B UD-Q2_K_XL | 41.8 GB | 4.42 GB | 37.4 GB, ~3.0 MB a record (IQ2_XS gate/up, IQ3_XXS down) | 48 x 256 x top-8, embd 3072 | none known; small context |
| Qwen3.5-122B-A10B UD-IQ1_M / UD-IQ3_XXS / Q3_K_S | 34.2 / 44.7 / 52.5 GB | 4.31 / 5.17 / 6.28 GB | 29.9 / 39.6 / 46.2 GB | same | dense grows with quant |

(Sizes from the GGUF headers fetched by HTTP range, `~/paddock-runs/w/big/gguf_head.py`; split GGUFs are supported by `MappedGguf`; every i-quant type here is in `kq_params`.) Plan: 122B UD-Q2_K_XL first (disk path as is), then the 397B with a dense-weights-on-host change. Download in progress: `~/paddock-runs/models/q122/` (resumable: `~/paddock-runs/w/big/dl.sh <url> <dest>`).

## Serving recipe (the Pi model on asus: Qwen3.6-35B-A3B, pinned expert offload, 32K context)

Build (in the checkout, dev shell from PR 1; about 2.5 min):
```
cd ~/paddock && nix develop ~/paddock-devshell --command cargo build --release --locked -p paddock-runner --no-default-features
```
Binary: `~/paddock/target/release/paddock-runner` (68 MB, built from asus/ssd-expert-backing 3334408; `--no-default-features` drops PDF rendering, which needs the PDFium archive). Kernel pack: `/nix/store/agnxcsrh0jbx4vbqkg6kwg4zzpi3jm0p-paddock-cuda-sm86-0.1.21/lib/paddock/pd-cuda-sm86.so` (the PR 1 flake's `packages.x86_64-linux.paddock-cuda-sm86`).

Files:
- Model: `/home/lukas/paddock-runs/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, 22360456160 bytes (20.8 GiB), sha256 `707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450`.
- Disk artifact (only for the disk-backed serve, not the Pi model): `/home/lukas/paddock-runs/paddock-ssd-art/qwen36-35b-a3b-ud-q4kxl.pdxa`, 20669863152 bytes (19.25 GiB); written from the GGUF on the first disk-backed start if missing (about 70 s).

Config (`paddock.toml`):
```toml
host = "127.0.0.1"
port = 11540
model = "/home/lukas/paddock-runs/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
max_ctx = 32768
max_batch = 1
vram_budget = 7500
prefill_chunk_rows = 1024

[moe_offload]
enabled = true
```

Command (as on asus today, a user unit; the Nix service replaces the systemd-run wrapper):
```
systemd-run --user --unit=paddock-serve --collect -p MemoryMax=26G -p MemorySwapMax=0 \
  -E LD_LIBRARY_PATH=/run/opengl-driver/lib -E RUST_LOG=info \
  ~/paddock/target/release/paddock-runner --config <dir>/paddock.toml \
  --kernel-pack /nix/store/agnxcsrh0jbx4vbqkg6kwg4zzpi3jm0p-paddock-cuda-sm86-0.1.21/lib/paddock/pd-cuda-sm86.so
```
`LD_LIBRARY_PATH=/run/opengl-driver/lib` is required (cudarc dlopens libcuda.so by name). No dev variables.

Resources (measured on the running server): VmRSS 19.8 GiB (18.3 GiB of it the page-locked expert planes, RssAnon 378 MiB); cgroup memory.peak 26.0 GiB with `MemoryMax=26G` (the rest is page cache of the GGUF, reclaimable; the cap is what holds it); VRAM 6907 MiB of 8192 (14 expert slots a layer at 32K). Start-up: about 23 s from start to `admin surface listening` (load 20 s + batched warm wave 2.3 s); the API answers on `http://127.0.0.1:11540` (OpenAI + Anthropic). GPU shared with ollama: stop ollama models first (`ollama stop`).

Measured on it: decode 27-30 tok/s, TTFT 0.8 s short prompts, 2.8 s at 300 tokens, 26.7 s at 6199 tokens cold and 0.44 s when the same prefix repeats; tool calls well-formed.

Pi client settings that keep long tasks moving on this serve (measured, see "Pi on the pinned serve"): in Pi's `models.json` the model entry `{"id": "Qwen3.6-35B-A3B-UD-Q4_K_XL", "contextWindow": 32768, "maxTokens": 4096, "reasoning": false, "samplingParams": {"chat_template_kwargs": {"enable_thinking": false}}}` under an `openai-completions` provider; in `settings.json` `"compaction": {"reserveTokens": 6144, "keepRecentTokens": 8000}`; short tool output (grep cargo output, read files in ranges).

Disk-backed variant (bounded RAM, 4K context): same binary, `max_ctx = 4096`, `vram_budget = 7500`, and in `[moe_offload]` add `disk_artifact = "<artifact>"` and `verify_once = true` (our serving config; the code default is off); `MemoryMax=8G`. Decode 9-15 tok/s.

## Pi on the pinned serve (real coding tasks, 2026-11-25)

Pi 1.2.0 (`/etc/profiles/per-user/lukas/bin/pi`) against `http://127.0.0.1:11540/v1`, scratch agent dir `~/paddock-runs/w/pi/agent` (`models.json`: provider `paddock`, `api: openai-completions`, model `Qwen3.6-35B-A3B-UD-Q4_K_XL`, `contextWindow 32768`, `maxTokens 4096`), print mode, thinking off, run inside the dev shell for cargo, in a scratch clone of sammasak/llm-gateway (`~/paddock-runs/w/pi/llm-gateway`, nothing pushed). Pi print mode hangs on an open stdin: run it with `< /dev/null`.

- Task 1, "add a unit test `ceil_secs_rounds_up` for `Millis::ceil_secs` in crates/core/src/time.rs and run the time tests": done in 63 s, 4 turns, 3 tool calls (read, edit, bash `cargo test`), correct test (0, 1, 1000, 1001 ms), only that file changed, `3 passed`.
- Task 2, "a test fails (regression_70_telemetry_privacy_and_exporter_isolation, unwrap on None at issue_70.rs:361): find the cause and fix it": not finished at the 40-minute timeout. 46 turns, 46 well-formed tool calls (read, grep, cargo test), 2 benign tool errors (greps with no match), no file changed. It got as far as the right lead: the test passes alone and fails in the full suite (shared tracing subscriber state between tests). Where it broke: the 32K window. Pi auto-compacted 7 times (one compaction failed: "generation hit the token cap"); each compaction and the turn after it re-prefill ~20-25K tokens cold at ~230 tok/s (2-5 min a turn), while the 40 of 46 turns that hit the server's prefix cache took 6-10 s. 10807 output tokens over 2157 s.
- So the tool loop holds (well-formed calls every turn, no malformed JSON, no lost turns); long tasks are throughput-bound by cold prefill after compaction at 32K.
- Fable consult on task 2's transcript: compactions rebuilt a 20-25K context each time (Pi's default keepRecentTokens 20000), so six consecutive compaction turns took 51% of the run; thinking was still on (about 8K of the 10.8K output tokens, and the cause of the failed summary); three tool outputs of 25-53K characters each triggered a compaction. Fixes, all Pi-side config (`~/paddock-runs/w/pi/agent`): `settings.json` `"compaction": {"reserveTokens": 6144, "keepRecentTokens": 8000}` and `"modelThinkingLevels": {"paddock/Qwen3.6-35B-A3B-UD-Q4_K_XL": "off"}`; `models.json` model entry `"reasoning": false, "samplingParams": {"chat_template_kwargs": {"enable_thinking": false}}` (checked: 3 output tokens for "17 times 3", no reasoning block); `--append-system-prompt` asking to pipe cargo output through `grep -E 'error|warning|panicked|FAILED|test result' | head -80` and read files in ranges of at most 150 lines. Server unchanged (a 64K window costs about 4 slots and makes each compaction longer; prefill_chunk_rows 1024 stays).
- Task 2 again with those settings: 902 s, 42 turns, 41 well-formed tool calls, 0 tool errors, 1 compaction (was 7). It reached the root cause: the test installs its capture subscriber with `tracing::subscriber::set_default`, which is thread-local, while the async request runs on other runtime threads. Then it stopped calling tools and deliberated in plain text about `with_default`, `block_on` and `spawn_blocking` until the 4096-token output cap (`stopReason: length`), ending without an edit. So with the settings above the loop is fast enough; what limits a deep task is the model (3B active) drifting into long text instead of trying a fix. Pi's final-turn length stop is where it breaks.

## Resume here (state at 2026-11-25 16:30, branch asus/ssd-expert-backing)

- 2026-11-25 about 18:45: the hand-started serve (user unit `paddock-serve`) was replaced by the declared system service (nixos-config PR 817); the user-unit lines below are history.

- Goals done: SSD streaming (accepted), HTTP serving of the disk-backed 35B (queue item 6), model plan for the Pi agent (#807, first candidate measured).
- Running on asus until 20:45: `paddock-serve.service` (user unit), the 35B **pinned** at max_ctx 32768 on 127.0.0.1:11540, config `~/paddock-runs/w/serve-pinned/paddock.toml`. Restart: `~/paddock-runs/w/serve-pinned/start.sh` (26G cap). The disk-backed serve: `~/paddock-runs/w/serve/start.sh` (8G cap). Requests: `python3 -I ~/paddock-runs/w/serve/stream.py <request.json>` (chat.json, tool.json, long*.json).
- Next step: real Pi tasks through the gateway against the pinned serve (coordinator; needs a non-loopback bind and an API key, which is the owner's call). If RAM must stay bounded instead: a chunked serial prefill for the disk path (it is limited to max_ctx 4096 on 8 GB), then the Q3_K_XL quant.
- Owner decisions open: default of `verify_once` (off) and of the host tier (off); whether to file the K-split capture report (no, per the coordinator).
- Builds: `nix develop ~/paddock-devshell --command cargo build --release --locked -p paddock-runner --no-default-features` (2.5 min); GPU gates: `~/paddock-runs/w/run.sh <name> <cap> <bin> <test>`.

## Strongest model for the Pi agent on asus (nixos-config#807)

Planned with a Fable consult (read-only, repo + web). Candidates:

| model / file | disk | path in Paddock | fits this host? | expected / measured | agent fit |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B UD-Q4_K_XL (on asus) | 22.4 GB | qwen35moe, pinned `[moe_offload]` (experts 18.3 GiB page-locked) | yes, 26G cap | measured below: 27-31 tok/s, 32K ctx | proven tool calls |
| same, disk-backed | + 20.7 GB artifact | qwen35moe, `disk_artifact` | yes, 8G cap | measured: 9-15 tok/s, 4K ctx only | same model, too slow and short for an agent |
| Qwen3.6-35B-A3B UD-Q3_K_XL | 16.8 GB + ~14.5 GB artifact | qwen35moe, disk | likely | est. 16-20 tok/s on disk; 3-bit tool-call quality unmeasured | risk |
| Qwen3.8-Flash-Next UD-IQ1_S / Q4_K_XL | 66.4 / 111 GB | qwen4exp: pinned offload only (disk refused) | no: pinned needs > 32 GiB; a disk port is not built | est. 3-4 tok/s if ported | no |
| gpt-oss-20b, Nemotron 3.5 30B-A3B, Laguna XS, Gemma 4 26B-A4B | - | VRAM-resident only | no (8 GB) | - | out |
| Qwen3.5-9B dense | ~6 GB | qwen35, resident | yes | fast, small ctx | weaker fallback |

Result: nothing larger than the 35B is servable on this host today; the strongest agent setup is the 35B Q4_K_XL served **pinned** (using the 32 GiB of RAM), not disk-backed. Measured on asus (`~/paddock-runs/w/serve-pinned/`, `start.sh` under `MemoryMax=26G`, swap off, memory.peak 26.0 GiB), config only:

```toml
host = "127.0.0.1"
port = 11540
model = "/home/lukas/paddock-runs/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
max_ctx = 32768
max_batch = 1
vram_budget = 7500
prefill_chunk_rows = 1024   # new key: the default 8192-row chunk leaves no VRAM for the slot cache

[moe_offload]
enabled = true
```

| config | slots | chat (23 in) TTFT / decode | tool call (302 in) TTFT / decode | 194-204 out decode |
|---|---|---|---|---|
| max_ctx 16384, default chunk (elected 4096) | 0 (zero-copy) | 0.76 s / 15.5 tok/s | 2.93 s / 20.0 | 15.5 |
| max_ctx 16384, chunk 1024 | 16 | 0.77 s / 30.0 | 2.80 s / 31.1 | 28.0 / 29.0 |
| max_ctx 32768, chunk 1024 | 14 | 0.76 s / 28.7 | 2.81 s / 29.7 | 26.8 / 27.8 |

Tool call in every row: `finish_reason=tool_calls`, `get_weather {"city":"Stockholm"}`. Long prompt at 32K ctx: a 6199-token prompt (a Rust source file, "what does this module do?") answered correctly, TTFT 26.74 s (232 tok/s prefill), decode 22.4 tok/s; the same prompt again: TTFT 0.44 s (prefix cache). Pinned serving runs the batched lane at width 1 (prefix cache, chunked prefill, decode pipe); the K-split capture bug did not show.

Next for #807: real Pi tasks through the gateway (coordinator). The server listens on 127.0.0.1:11540 only; the gateway needs a host/key decision.

## HTTP serving on asus (queue item 6, nixos-config#508)

Config only, no dev variables (`~/paddock-runs/w/serve/paddock.toml`; started by `w/serve/start.sh` under `systemd-run --user -p MemoryMax=8G -p MemorySwapMax=0`):

```toml
host = "127.0.0.1"
port = 11540
model = "/home/lukas/paddock-runs/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
max_ctx = 4096
max_batch = 1

[moe_offload]
enabled = true
disk_artifact = "/home/lukas/paddock-runs/paddock-ssd-art/qwen36-35b-a3b-ud-q4kxl.pdxa"
# host_cache_gb = 4   # optional pinned tier of hot experts
```

Startup log: `disk-backed slot cache seated slots=24`, then `serving SERIAL by design - disk-backed MoE experts ([moe_offload] disk_artifact) serve one sequence on the eager serial loop`. With max_ctx 8192 the start refused cleanly: the 8192-row prefill scratch is 3.96 GB and `the budget buys 0` slots (needs 8). `max_batch > 1` keeps the startup refusal "batched serving (enable_batch) is not supported with them yet - serve this model with pinned backing for it".

Requests (`w/serve/stream.py`, streaming, temperature 0, thinking off via `chat_template_kwargs`):
- Chat, "In two sentences, explain why the sky is blue.": "Sunlight reaches Earth's atmosphere and is scattered in all directions by all the gases and particles in the air. Blue light is scattered more than other colors because it travels as shorter, smaller waves, which is why we see a blue sky most of the time." 23 prompt tokens, 53 out, TTFT 2.29 s, decode 8.54 tok/s. (With thinking on, the same prompt gives a correct Rayleigh-scattering reasoning trace.)
- Tool call, one function `get_weather(city, unit?)`, "What is the weather in Stockholm right now? Use the tool.": `finish_reason=tool_calls`, `get_weather {"city":"Stockholm"}` (streamed and non-streamed, id `call_...`), 302 prompt tokens, TTFT 11.97 s, decode 7.38 tok/s. A cold prefill of a few hundred tokens touches most of each layer's experts, so it reads a large part of the artifact (1.3 GB/s device).

Serving tuning (8G cap, `w/serve/ab.sh`: artifact page cache dropped before each start, then 4 requests of ~195 tokens out, thinking off). Without the drop, page cache left by earlier processes is served past the new cgroup's cap and inflates decode (one such run read 16.9-19.4 tok/s).

| config | slots | decode, 4 requests | TTFT |
|---|---|---|---|
| default VRAM budget (90% of the card) | 24 | 9.42 / 10.63 / 10.73 / 10.98 tok/s | 4.39, then 3.2-3.4 s |
| `vram_budget = 7500` | 30 | 10.14 / 11.34 / 11.34 / 11.64 | 4.30, then 3.3-3.5 s |
| `vram_budget = 7500`, `host_cache_gb = 3` | 30 | 9.40 / 11.08 / 11.63 / 8.31 | 4.25, then 3.2-3.6 s |

So `vram_budget = 7500` (+6%) is in the serve config; the host tier does not pay under an 8 GiB cap (it takes the page cache's room).

Decode-miss cost (Fable consult, read-only): the per-read BLAKE3 check of each 1.9 MB miss sits on the critical path. `[moe_offload] verify_once = true` (default off; the owner's call) checks a record on its first read in the process only. Serve A/B (`ab.sh`, 30 slots, cache dropped): off 10.21 / 11.35 / 11.35 / 11.68 tok/s, on 12.02 / 14.32 / 15.08 / 15.12 tok/s. Gate with it on (R26): parity bit-exact as before, 9246 of 54545 reads checked a digest, gate decode 11.20 / 11.55 tok/s (9.7 before). The consult ranked page-cache prefetch from the previous token's routing as no gain (a miss is by construction an expert not routed recently) and a frequency-aware VRAM policy as +5-10%, uncertain.

Final serve config (`w/serve/paddock.toml`: the block above plus `vram_budget = 7500` and `verify_once = true`), artifact page cache dropped, fresh start, 8G cap:

| request | result | TTFT | decode |
|---|---|---|---|
| chat, sky blue (23 in, 53 out) | the correct two-sentence answer | 3.62 s | 9.36 tok/s |
| tool call `get_weather` (302 in, 27 out) | `finish_reason=tool_calls`, `{"city":"Stockholm"}` | 12.28 s | 8.71 tok/s |
| hash map paragraph (31 in, 194 out) | coherent | 3.18 s | 11.86 tok/s |
| B-tree paragraph (32 in, 199 out) | coherent | 3.22 s | 13.56 tok/s |

Limits seen: max_ctx 8192 does not fit (prefill scratch for the whole prompt, 3.96 GB); a cold prompt of a few hundred tokens reads most of the artifact (TTFT about 12 s at 300 tokens).

Review of the serve change (one subagent pass): serial fallback, non-disk behaviour and verify_once sound. Fixed after it: `max_batch > 1` with `disk_artifact` now refuses before any load: "[moe_offload] disk_artifact serves one sequence on the serial loop, but max_batch = 2: set max_batch = 1, or drop disk_artifact to serve with pinned backing" (checked on asus); a stale `<artifact>.partial` names itself and the way out; Metal refuses disk_artifact; `enabled = false` with disk_artifact warns; an unknown VRAM headroom is an error; the decode pipe is off on disk. Open: no parent-directory fsync after the artifact rename; image requests on an mmproj model get only the 512 MiB margin.

## Current state (head of asus/ssd-expert-backing), against the PR 2 checklist

Each line points at the run below that proves it; logs are in `~/paddock-runs/w/logs` on asus.

- Pinned path bit-identical: R3 `kq_moe_pair_host_mapped_bitmatches_resident` bitwise true on the branch; `gpu_qwen36_moe_host` greedy tokens byte-equal to main in 10 of 10 interleaved runs (R4b, R4c, R4d, R4e).
- Pinned path no slower: R4f (a061286, the head's code) decode main 31.12, 31.08, 31.08, 31.09, 31.07 (median 31.08); branch 31.12, 31.08, 31.09, 31.07, 31.08 (median 31.08); same hit rate 70.5%; tokens byte-equal 10 of 10; R3 bitwise true.
- Disk parity: prefill logits bit for bit equal to pinned's default class and greedy tokens equal, for a one-token, a 37-token text and a 600-token synthetic prompt (sorted class, waves, eviction, duplicate routed ids across tokens), in every process of the full R7 at 2bc49a9. A 5-token prompt (LIST waves, under the sorted threshold) is teacher-forced 32/32 with equal greedy tokens.
- Bounded RAM: the full gate passes under a 2 GiB cgroup cap with swap off (RssAnon 287 MB) against a 19.25 GiB artifact; no OOM, no Xid.
- Throughput at 4K (4000-token prompt, one request), full R7 at 2bc49a9: 8G cap cold TTFT median 14.40 s (n=5), warm 13.26 s (n=20), decode 9.72 tok/s; 2G cap 14.35 s, 14.12 s, 5.45 tok/s. Pinned on the same harness (R22, 28G): TTFT 13.23 s, decode 34.45 tok/s.
- I/O: miss rate, artifact read rate and device read_bytes per run are in every R7 log line (prefill reads median 1420 MiB/s at 8G, the SSD's measured ceiling is 1.35-1.65 GB/s).
- Clean failure: foreign artifact (R9b), fewer slots than top-k, DFlash, graph replay and batched serving (R15) refuse with a sentence naming the cause and the way out; a dense qwen35 ignores the switch (245c57c). Not covered on the GPU: a corrupt record (CPU tests cover it; it fails the request and invalidates the slots, it is not detectable at load without reading 19 GiB).
- Stability: R24 on the head's code (a061286, 8G cap): 26 min 9 s, 1 cold + 80 warm runs of a 4000-token prefill and 64 decode tokens, all ok; warm TTFT median 11.53 s (9.82-20.92), decode median 9.72 tok/s (4.21-9.87), 1155331 expert reads, 2.33 TB from the artifact (1.30 TB from the device), RssAnon 287 MB, GPU flat at 7271 MiB, NVMe peak 44.9 C, no Xid/NVRM/oom. Five runs over 15 s, all with slow reads (open question). Earlier R16 (ddf764c) was clean too.

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

- Dev shell: `asus/devshell-driver-lib` (one commit, 355df36, on top of PR 1's codex/hpc-research) adds a shellHook that puts /run/opengl-driver/lib on LD_LIBRARY_PATH. Checked: `env -u LD_LIBRARY_PATH nix develop . --command` prints `LD=/run/opengl-driver/lib`; nixfmt, statix and deadnix pass on flake.nix. To be folded into PR 1.
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

### Overlap and staging (cd590a3, ffdcc30)

Each staging batch's copies are followed by a CUDA event; a fill waits only for the oldest batch when it needs a buffer, and routing goes up from two pinned buffers, so a wave's reads overlap the previous wave's kernels. Staging pool sweep (8G, 32 slots, 4 readers, run 0 cold / run 1 warm TTFT): 8 records 26.6 / 25.9 s, 32 records 20.5 / 20.2 s, 64 records 20.2 / 20.1 s, 128 records 20.1 / 20.1 s. Default is now one wave (the slot count, capped at 64 records; 32 records is about 61 MB pinned). Reviewed by a second code-reviewer subagent: staging path sound; its findings (a failed event wait dropped an upload buffer's guard, no test for two batches in flight or a failed mark, a panicked reader skipped invalidation, uncapped staging default) are fixed in ffdcc30.

### Full R7 at 3fc49a8 (5 processes x 1 cold + 4 warm, 4000-token prompt, 64 decode tokens, 32 slots, swap off)

Command: `~/paddock-runs/w/r7full.sh 3fc49a8 <cap>` (logs `~/paddock-runs/w/logs/R7full-3fc49a8-<cap>-p*.log`), summarized with `w/summ.py`.

| cap | cold TTFT (n=5) | warm TTFT (n=20) | decode cold run (n=5) | decode warm (n=20) | prefill read |
|---|---|---|---|---|---|
| 8G | median 20.54 s (20.46-20.58) | median 20.31 s (20.24-28.36) | 9.53 tok/s (9.44-9.54) | 9.60 tok/s (9.51-9.68) | median 1415 MiB/s (803-1430) |
| 2G | median 20.43 s (20.40-20.48) | median 20.43 s (20.36-20.52) | 5.61 tok/s (5.59-5.63) | 5.55 tok/s (3.30-5.60) | median 1309 MiB/s (1278-1333) |

- Parity passed in all 10 processes (one-token bit exact; text and synthetic teacher-forced 32/32, free-running equal).
- Bounded RAM: cgroup memory.peak equals the cap (page cache is charged); at 2G the process holds RssAnon 286 MB and RssFile 2.7 GB (mmapped GGUF, reclaimable) at the end, against a 19.25 GiB artifact. No swap (MemorySwapMax=0), no OOM kill, no Xid (`journalctl -k --since -2h` has no NVRM/Xid/oom lines).
- GPU peak 7271 MiB of 8192.
- Outliers: at 8G two warm runs (p1 and p2, run 1) took 27.9 and 28.4 s with reads at 818 and 803 MiB/s; one 2G warm decode at 3.30 tok/s. Cause not established (nothing else of mine ran then).
- Against pinned on the same harness (R8: TTFT 149.1 s, decode 34.3 tok/s at 28G): disk prefill is 7x faster (pinned prefill reads experts zero-copy over PCIe per token), disk decode is 3.6x slower at 8G and 6x at 2G (read bound).
- R4c (pinned path at 3fc49a8): tokens byte-equal to main in 10 of 10 runs; decode median 31.11 on both (main 31.08-31.13, branch 31.10-31.13); R3 bitwise true.

### Stability (R16, ddf764c, 8G cap, 32 slots)

One process, 1 cold + 55 warm runs of a 4000-token prefill and 64 decode tokens, 26 min 20 s of service runtime: all ok. Warm TTFT median 20.08 s (19.75-27.99), decode median 9.59 tok/s (9.27-9.78), 808069 expert reads, 1.63 TB from the artifact (987 GB from the device). RssAnon 286 MB at the end, GPU memory flat at 7271 MiB, no swap, no Xid/NVRM/oom in `journalctl -k`. Five runs took about 27.8 s with slow reads (runs 2, 24-25, 43-44: pairs about 10 minutes apart); the cause is not established. NVMe temperature is now sampled by the run helper; in later runs it peaked at 43.9 C (max 74.9 C).

### Read size (negative result)

fio on the artifact, direct random reads, 12 s each: 512 KiB 956 MiB/s (1 job) / 1083 (4 jobs); 1 MiB 1148 / 1144; 2 MiB 1346 / 1280; 4 MiB 1652 / 1366. Tried coalescing adjacent expert records (waves take experts in id order) into one `preadv` of up to N records, buffered (R17, 8G, 32 slots, cold/warm TTFT): 1 reader 1 record 21.0 / 20.7 s; 4 readers 4 records 20.7 / 20.5 s (decode 9.96 / 10.06 tok/s); 1 reader 8 records 21.1 / 20.7 s; 2 readers 8 records 21.7 / 21.1 s; 1 reader 32 records 21.1 / 20.8 s. Buffered reads stay at 1.1-1.4 GB/s whatever the run length, so the change was reverted (not committed). Large reads would need direct I/O, which needs 4 KiB-aligned records (an artifact format change).

### Refusals and environment (ddf764c)

- `moe_disk_refuses_what_it_cannot_serve` (R15): 4 slots at top-8 refuse with "need a VRAM slot cache of at least 8 experts a layer (top-8 routing); the budget buys 4"; DFlash, `bench_decode_ms` (graph replay) and `enable_batch` refuse with "is not supported with them yet - serve this model with pinned backing for it"; afterwards the model generates pinned's tokens [2614, 513, 5081, 5576]. `moe_disk_refuses_a_foreign_artifact` (R9b) passes on the same binary.
- Dev shell (asus/devshell-driver-lib): with LD_LIBRARY_PATH unset outside, `nix develop ~/paddock-devshell` sets it and `kq_moe_pair_host_mapped_bitmatches_resident` runs (`gate_up bitwise=true down bitwise=true`). Outside the shell with no driver path, the heavy disk gate fails: `gate cannot run: no usable CUDA device (...)`, `test result: FAILED`.

### Decode design consult (Fable subagent, read-only, 2026-11-25)

Question: which change raises decode tok/s most at a bounded host RAM budget. Answer, ranked: (1) an explicit pinned host tier of hot expert records, admitted by routing frequency (counts from the ids `plan` already downloads), copied straight from the tier to the slot with no read and no second digest check: estimated 14-16 tok/s at 6-8 GiB from 9.6, because even a page-cache hit costs a 2 MB memcpy plus BLAKE3 (the whole artifact in page cache still caps decode at 13.3 tok/s); (2) not allocating the sorted path's partials scratch on the disk path, about 3 more slots, +4-6%; (3) a frequency-aware VRAM policy, deferred because disk and pinned would then see different misses; (4) no legal overlap inside a decode step without speculation. Next step follows (1).

### Host tier (e6f4d75; design from the Fable consult above)

A pinned host tier of expert records, split evenly between layers, admitted by routing count (an expert replaces the least routed held one when routed at least as often). A slot miss found there is copied straight to the slot, with no read and no second digest check; a tier buffer is never written while a copy out of it is pending (it rides the batch marks like staging). Off unless `PADDOCK_MOE_DISK_HOST_MB` is set. Parity unchanged (all R18 runs: one-token bit exact, text and synthetic teacher-forced 32/32, free-running equal).

R18 (32 slots, 4000-token prompt, 64 decode tokens; run 0 cold, runs 1-2 warm; "tier" = slot misses served from the tier out of 5720 per decode):

| cgroup cap | tier | cold / warm TTFT | decode run 0 / 1 / 2 | tier share of decode slot misses | without a tier |
|---|---|---|---|---|---|
| 4G | 2 GiB | 20.1 / 20.1 / 20.0 s | 6.79 / 3.66 / 6.66 tok/s | 1150 | 6.5-6.8 tok/s |
| 8G | 4 GiB | 20.0 / 19.8 / 19.9 s | 10.56 / 13.34 / 13.29 | 1990 | 9.6 |
| 8G | 6 GiB | 19.8 / 19.7 / 19.7 s | 9.92 / 11.29 / 11.74 | 2730 | 9.6 |
| 16G | 8 GiB | 19.5 / 19.5 / 19.5 s | 16.28 / 16.51 / 16.62 | 3360 | 11.9 |
| 16G | 12 GiB | 19.2 / 19.2 / 19.2 s | 17.56 / 18.41 / 18.68 | 4470 | 11.9 |

The tier is charged to the cgroup like the page cache, so it pays when it leaves room for page cache (4 of 8 GiB; 6 of 8 GiB is worse than 4) and not at all at 4G. Peak VmHWM rises by the tier size (7.4 GiB with 4 GiB). Not done from the consult: reclaiming the sorted path's partials scratch on the disk path, because the shared expert's 2-64-row path borrows it for K-split planes and its size need there is not established.

### Host tier, interleaved A/B (af00dbb, `w/tier-ab.sh`, 3 processes each, runs 1-2 warm)

The single-process R18 numbers above overstated the gain: the page-cache share of decode reads varies between processes (the second 8G/4 GiB process gave 10.84 against 13.34). Interleaved, 32 slots:

| cap | tier | warm TTFT | warm decode |
|---|---|---|---|
| 8G | none | median 20.31 s (20.27-28.11, n=6) | median 9.60 tok/s (9.55-9.63) |
| 8G | 4 GiB | median 19.89 s (19.84-19.91) | median 10.79 tok/s (10.68-10.87) |
| 16G | none | median 20.20 s (20.12-20.31) | median 14.98 tok/s (14.89-15.61) |
| 16G | 8 GiB | median 19.50 s (19.49-19.52) | median 16.69 tok/s (16.55-16.96) |

So the tier gives +12% (8G) and +11% (16G) decode and 0.4-0.7 s TTFT. The consult's estimate (+50%) assumed the single-reader page-cache numbers (11.9 tok/s at 16G); with four readers, page-cache hits are already cheaper. Parity passed in all 12 processes. Review findings on the tier (a buffer in two in-flight batches, stats consumers, unused remainder) are fixed in af00dbb with a mutation-checked test.

### Pinned path at af00dbb (R4d)

Tokens byte-equal to main in 10 of 10 runs, same hit rate (19840 rows, 5845 misses). Decode rerun: main 31.09, 31.11, 31.07, 31.06, 31.10 (median 31.09); branch 31.07, 31.06, 31.03, 31.07, 31.07 (median 31.07). Inside main's range, at its low edge; the pinned path's code did not change since R4c (medians 31.11 both). R3 bitwise true.

### Prefill in the pinned sorted class (a325550, 2bc49a9)

Profiling (temporary timing around the disk MoE call, artifact fully in page cache, 27G): the LIST pair took 17.0 s of an 18.9 s 4000-token prefill, with reads only 1.9 s. The pinned backing prefills in its sorted tensor-core class (`moe_align` + `kquant_moe_gate_up_mma` + `kquant_moe_down_mma` + `moe_slot_combine`). `pd_moe_align` cannot take a masked routing (it counts `idx[p]` with no absent check), so the host builds the same layout per wave from the pairs it already knows (`sorted_layout`: pairs grouped by slot into 32-row blocks) and uploads it into the sorted scratch. The unchanged kernels then write each pair's weighted partial once, over all waves, and the slot combine folds them in pair order as on the pinned path. A row of the sorted pair does not depend on its block neighbours (one CTA per 32-row block, no K-split across CTAs), so this is the pinned result bit for bit. The disk path takes it exactly when the pinned path takes the k-quant sorted branch (same `sorted` decision, one gate/up type, no i-quant seat); short prompts under the sorted threshold with more experts than slots keep the LIST waves.

The gate now runs the pinned reference in its default class (no `PADDOCK_NO_SORTED_QMOE`). R21-disk (8G, 32 slots):

```
parity one token (1 tokens): 0 of 248320 prefill logits differ bitwise, |d|/rms 0.00000, greedy tokens equal=true
parity five tokens (5 tokens): 248320 of 248320 prefill logits differ bitwise, |d|/rms 0.01383, greedy tokens equal=true  (LIST waves; teacher-forced top-1 32/32)
parity text (37 tokens): 0 of 248320 prefill logits differ bitwise, |d|/rms 0.00000, greedy tokens equal=true
parity synthetic (600 tokens): 0 of 248320 prefill logits differ bitwise, |d|/rms 0.00000, greedy tokens equal=true
```

TTFT at 4000 tokens (R21-disk2, 8G, 32 slots): cold 14.40 s (reads 12.21 s at 1273 MiB/s), warm 13.30 / 13.25 s; decode 9.55 / 9.73 / 9.75 tok/s.
Pinned on the same harness in its default class (R22, 28G cap, graphs): TTFT 13.24 / 13.23 / 13.23 s, decode 33.44 / 34.47 / 34.45 tok/s, VmHWM 29.3 GiB.

So disk-backed prefill now runs at the pinned speed (warm within 0.1 s, cold +1.2 s) under an 8 GiB cap, bit for bit equal; decode is 3.5x slower (read bound). The earlier pinned baseline (R8, 149 s TTFT) ran the token-batched class the old harness forced.

### Full R7 at 2bc49a9 (5 processes x 1 cold + 4 warm, 4000-token prompt, 64 decode tokens, 32 slots, swap off)

| cap | cold TTFT (n=5) | warm TTFT (n=20) | decode cold run (n=5) | decode warm (n=20) | prefill read |
|---|---|---|---|---|---|
| 8G | median 14.40 s (14.33-22.48) | median 13.26 s (12.92-21.21) | 9.50 tok/s (9.45-9.63) | 9.72 tok/s (9.58-9.90) | median 1420 MiB/s (766-1471) |
| 2G | median 14.35 s (14.31-22.50) | median 14.12 s (13.97-22.53) | 5.43 tok/s (5.41-5.44) | 5.45 tok/s (3.24-5.48) | median 1304 MiB/s (765-1324) |

All 10 processes: one-token, text and synthetic prompts bit for bit equal to pinned, the five-token prompt (LIST waves) teacher-forced 32/32, exit 0. RssAnon 287 MB, VmHWM 3.05 GiB (mmapped GGUF), no swap, no Xid/NVRM/oom in `journalctl -k`, NVMe peak 45.9 C. The maxima near 22 s are the intermittent slow-read runs (about 765 MiB/s). Review of the sorted waves (subagent): no critical finding; its re-seat capacity check and `sorted_layout` test are in the commit after 2bc49a9.

Decode ceiling: about 90 slot misses a token at 32 slots (71.5% hit), about 2 MB each, so decode is bound by the device (1.3-1.4 GB/s) plus page cache; without speculative prefetch (out of scope per the RFC) only more host memory (page cache or the host tier) or more VRAM slots raise it. The sorted prefill now uses the `part` scratch, so reclaiming it for slots is no longer possible.

### Slot sweep on the head's code (R25, a061286, 8G cap, run 0 cold / runs 1-2 warm)

| slots | cold / warm TTFT | decode warm | decode slot hit | GPU peak |
|---|---|---|---|---|
| 16 | 15.03 / 13.73 / 13.66 s | 7.76 / 7.81 tok/s | 46.0% | 6055 MiB |
| 24 | 14.74 / 13.18 / 13.16 s | 9.50 / 9.50 | 63.4% | 6663 MiB |
| 32 (R21-disk2) | 14.40 / 13.30 / 13.25 s | 9.73 / 9.75 | 71.5% | 7271 MiB |
| 36 | 14.30 / 13.28 / 13.23 s | 10.02 / 10.01 | 74.6% | 7591 MiB |
| 40 | `prefill: Gpu(OutOfMemory)` in the parity prompts (a dev pin skips the VRAM budget) | | | |

Prefill stays bit for bit equal to pinned at every slot count (the wave grouping changes, the per-pair partials do not). Prefill time barely depends on slots; decode gains about 3% from 32 to 36.

Step 5 (config) waits on a serving lane: the runner serves through `enable_batch`, which the disk backing refuses at startup, so a `[moe_offload]` disk option would only move that refusal into config. The LIST waves stay because i-quant seats have no tensor-core lane (the 180B target is likely i-quant).

## Next

1. Owner decisions: keep or drop the host tier (e6f4d75, af00dbb) in the upstream stack; file the upstream K-split capture report (draft below); fold `asus/devshell-driver-lib` into PR 1.
2. Step 5 (config and estimator) is not started: slots, staging, readers and the host tier are development variables (`PADDOCK_MOE_CACHE_SLOTS`, `PADDOCK_MOE_DISK_STAGING`, `PADDOCK_MOE_DISK_READERS`, `PADDOCK_MOE_DISK_HOST_MB`).
3. Decode is read bound (see the ceiling note above).
4. The stack carries fix commits on top of the features (no force push here); squashing into the RFC's step order is for whoever rebases it for upstream.

## Open questions

- Host tier: worth +11-12% decode for about 250 lines (with tests)? Owner's call for the upstream stack.
- Intermittent slow reads (about 800 MiB/s instead of 1.4 GB/s for a whole prefill or decode, a few runs per half hour): the drive itself. A 12-minute fio soak on the artifact with nothing else running (`~/paddock-runs/w/fio-soak.sh`: 2 MiB random direct reads, 4 jobs, per-second log) averaged 1174 MiB/s, median second 1279 MiB/s, with 98 of 720 seconds under 1000 MiB/s: stretches of 13-14 s at about 530 MiB/s starting at t = 23, 89, 328, 423 and 531 s. NVMe peak 46.9 C, so not thermal; likely the DRAM-less A2000's firmware. Not a backing defect; it sets the spread of cold numbers on this host.

- Upstream pack bug: K-split router arm returns error 2 under capture with no partials plane. Draft an issue for truespar/paddock (owner decision to file).
- Why the 4000-token pinned capture does not hit it.
- Bounded RAM criterion: cgroup memory.peak counts page cache, so with buffered reads it reaches any cap; 4G works (RSS 3.0 GiB). Direct I/O would make the bound exact but loses the page-cache tier that lifts decode from 6.5 (4G) to 11.9 tok/s (16G).
- The parity prompt is synthetic (random ids) and the greedy output loops ([238, 197, 197, 1, 198, ...]); a real text prompt would make token parity a stronger check.

## Draft upstream report (not filed; filing on truespar/paddock is the owner's decision)

Title: qwen35 serial prefill capture fails with CUDA error 2 on dies under 128 SMs (K-split router scratch)

On an RTX 3070 (46 SMs, sm_86, pack 0.1.21), a 600-token `GpuQwen35::prefill()` with the experts on `PADDOCK_MOE_HOST` and `PADDOCK_NO_SORTED_QMOE=1` fails with `prefill: Gpu(Launch(2))`. `PADDOCK_PANIC_LAUNCH=1` names `batch_ops.rs:612` (`matvec_f32_raw`, through `matvec_f32_batch`).

Cause as read from the source: `pd_matvec_f32_batch` takes the K-split arm on dies under 128 SMs from 256 rows (`ks_small`). `pd_f32nt_ks_go` asks `pd_part_scratch` for a partials plane, which never allocates while the stream is capturing, and then returns `cudaErrorMemoryAllocation` when the stream has no plane big enough (`gemm/f32_qkv.cuh:584`). `prefill()` captures a graph the first time it sees a length, so the first prefill of such a length fails. The part_scratch comment says capture falls back to nz=1, but this arm returns an error instead.

Evidence: GPU memory peaked at 5.8 of 8 GiB. The same prefill passes with `PADDOCK_NO_PREFILL_GRAPH=1` or `PADDOCK_NO_F32NT_KS=1`. A 4000-token prefill on the same harness captures without error; why is not established.
