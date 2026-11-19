# Nix development and the first HPC experiment

Tracking: sammasak/nixos-config#508, with measurements in #493 and task correctness
in #494. This fork keeps upstream's native Rust/CUDA design. Nix packaging lives
here; host identities, ports, power policy and service assembly belong to
nixos-config. No Studio/manager packaging or engine rewrite is part of this slice.

## Reproducible development

The setup copies the standard `sammasak/rust-devenv-template` at
`0a019cf3e06939a5db6b8e1c79dce3f81ece56b8`. `devenv.yaml`/`devenv.lock` pin the
platform and toolchain overlay. The flake uses the same revisions, checked by
`checks.x86_64-linux.input-parity`. The upstream `rust-toolchain.toml` stays on
stable; the locked overlay fixes which stable compiler the Nix environments use.
Update both Nix/devenv pins together and rerun parity, rather than updating Rust
implicitly on shell entry. `nix/support.nix` shares native build dependencies and
the manifest-selected PDFium archive between development and packaging.

```sh
devenv shell
# Stage the upstream archive from its manifest URL and SHA256 via Nix.
devenv tasks run paddock:prepare
devenv tasks run paddock:nix-check
devenv tasks run rust:fmt-check
devenv tasks run rust:ci
# Packaging-only shell with the same compiler and native build tools:
nix develop
```

The CI task checks Nix formatting/static lint/unused bindings/input parity, Rust
format, workspace types, Clippy with warnings denied, dependency
policy, rustdoc with warnings denied, and workspace tests including doctests.
It uses upstream's default features: `--all-features` enables optional CUDA
forensics compilation and Metal integration, which need separate validation.
The upstream lint configuration and format policy are preserved. The broader
central tasks also expose audit, nextest, coverage, mutation and profiling tools;
a task being available is not evidence that it passed.

Rust hooks are explicitly deferred until the first actual CI baseline is green;
the TODO records the current blocker: the copied license policy rejects
`CC0-1.0` on `to_method 1.1.0` through rav1d. Bans and sources passed. Full
workspace CI is still pending. Do not enable a permanently failing hook
or describe this as full CI acceptance. After it is green, enable hooks and run
the standard adoption workflow's rejection probes. Do not change upstream code
or weaken dependency policy simply to make the first gate green.

## Native packages

```sh
nix build .#cargo-vendor --no-link
nix build .#paddock-runner
nix build .#paddock-cuda-sm86 --out-link result-cuda
nix flake check
```

The runner builds from `Cargo.lock`, including content hashes for Git dependencies.
The PDFium archive is a hash-checked Nix fetch from upstream's committed manifest;
there is no ad-hoc download in a derivation or service startup. Runner tests and
`--help` are part of its package gate. The runner does not compile the CUDA pack.

The pack uses CUDA 13.3, only architecture 86, and one nvcc build thread. Upstream
names the Linux output `pd-cuda-sm120.so` even for this target; this package installs
that ELF as `lib/paddock/pd-cuda-sm86.so`, rather than shipping the Windows-style
alias. Its install check reads the ELF and verifies sm_86 SASS with cuobjdump.
Compiling the pack needs a toolkit, not a GPU or loaded display driver. Running it
still needs the compatible NVIDIA driver and actual GPU validation.

## Existing measurements first

Read the actual runner `/metrics`; add only a proven missing measurement:

| Question | Existing evidence |
|---|---|
| Throughput, TTFT and decode timing | `paddock-bench`, server `gen_ai_server_*` histograms |
| Offered load and running work | `paddock_num_requests_in_flight`, `paddock_num_requests_running` |
| KV pressure and prefix reuse | `paddock_kv_cache_usage_perc`, `paddock_prefix_cache_{queries,hits}_total` |
| Tiering cost when explicitly enabled | `paddock_kv_tier_*` counters/gauges |
| GPU board energy, fan, heat and memory | Existing node GPU exporter plus monotonic 1 Hz trial recording |
| Useful coding capacity | Verified Pi tool actions/checks under #494; token rate alone is insufficient |

Do not infer scheduler queue latency by subtracting unrelated gauges or call TTFT
queue time. Inspect enqueue/admission timestamps first if queue delay is the
bottleneck. Any new histogram should use bounded operation/outcome labels, never
prompts, request/session IDs, model paths or tool payloads. Preserve missing and
unsupported hardware readings as unknown.

## First hardware feedback after 09:00

ASUS stays off until **20 November 2026, 09:00 Europe/Stockholm**. No auto-wake,
service activation, model download or default route change comes from this fork.
Keep the existing 130 W limit and measure one workload at a time.

1. Record source/package/kernel hashes, compiler/toolkit/driver, model digest,
   quantization, tokenizer, context, cache state and sampling settings.
2. Establish one-agent baseline, then two agents with continuous batching. Keep
   Ollama's model unloaded during the initial Paddock residency test.
3. Use the existing HTTP benchmark for end-to-end comparison and its in-process
   mode for kernel diagnosis. Its HTTP token-event count is not an authoritative
   server tokenizer count; client mode does not measure pure prefill throughput.
4. Test prefix reuse with both cold and repeated agent prefixes. Keep raw work
   private and separate token throughput from successful checked tasks/minute.
5. Require VRAM headroom, no Xid/OOM/thermal slowdown, acceptable p95 latency and
   sustained fan behaviour. Report GPU-board joules per successful task, with
   actual sample intervals and errors; this is not whole-host energy.
6. Only profile a demonstrated bottleneck: first a short Nsight Systems timeline,
   then a small Nsight Compute kernel selection. Keep profiled diagnostic runs
   separate from throughput baselines because collection changes runtime cost.

Primary research: [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
for architecture-specific resource limits; [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/)
for CPU/GPU timeline collection; [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
for collection overhead. These guide experiments; no performance gain is claimed.
