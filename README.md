# HELIX

A chunked associative scan (SSD) kernel for State-Space Models on Apple Silicon,
exposed as a drop-in custom operator.

Full architecture and roadmap: `~/.claude/plans/you-are-a-principal-compressed-wigderson.md`.

## What this is (and what it is not)

`llama.cpp` already ships a chunked SSD Metal kernel
(`kernel_ssm_scan_ssd_mma_f32`). HELIX is **not** "SSMs fall back to CPU on
Apple Silicon" -- that is no longer true. HELIX targets what upstream's fast
path leaves on the table:

| Upstream gate | HELIX |
|---|---|
| `d_inner == 64` required, else scalar fallback | any multiple of 8, via head-dim slabs across `grid.x` |
| scalar-`A` only (`ne30 == 1`) | diagonal-`A` planned (Week 4) |
| fp32 everywhere | bf16 operands via MPP on M5 (**done**, 4.0x on Pass C); fp32 elsewhere |
| chunk-state GEMM (`dS`) left scalar | on the matrix unit (**done**, 7.9x on that pass) |
| ragged tail punted to a scalar kernel | zero-padded, one GEMM path for every chunk (**done**) |
| serial cumulative decay on thread 0 | parallel shuffle scan (**done**) |
| one threadgroup per (head, seq), serial chunk walk | 3-pass chunk parallelism (**done**, 2.0x) |
| no M5 Neural Accelerator path | MPP `tensor_ops::matmul2d` (**done**) |

## Status

- **Milestone 1 -- reference + harness.** Done. fp64 sequential reference and an
  independent fp64 chunked reference agree to f32 round-off across 15 shapes
  including every sequence-length edge case. Mutation-tested: a 1e-7 relative
  perturbation is caught.
- **Milestone 2 -- fp32 `simdgroup_matrix` kernel.** Done. 30/30 parity against
  the fp64 reference on M5 Max, including `head_dim` 128 and 192 where upstream
  falls back entirely. Mutation-tested against 5 seeded kernel bugs.
- **3-pass chunk parallelism.** Done. Occupancy only -- still fp32, state still
  in device memory. **2.0x** at `seq=1`.
- **Chunk-state GEMM on the matrix unit.** Done. Pass A was 65% of runtime and
  entirely scalar; it is now a GEMM (`dS = B^T . dtXs`, folding the per-timestep
  decay into `dtX`). **7.9x on that pass, 2.3x overall.** Combined with the
  3-pass work: **4.69x** over single-pass at `seq=1`.
- **Milestone 3 -- reduced precision via MPP.** Done. Reframed after measurement:
  bf16 through `simdgroup_matrix` is 1.5x *slower* on M5, so bf16 and MPP are
  one change, not two. Pass C now runs on the M5 Neural Accelerators with bf16
  operands and fp32 accumulators: **4.0x on that pass, 1.58x overall**, for a
  cumulative **7.45x** over single-pass. `HelixMatmulTraits` picks it per device;
  M1-M4 keep the fp32 `simdgroup_matrix` path. See `docs/PRECISION.md`.
- **Milestone 4 -- llama.cpp integration.** Done. One `#ifdef` upstream (43-line
  diff, pinned SHA), passing ggml's own op test 13/13.
  - **Kernel head-to-head** vs `kernel_ssm_scan_ssd_mma_f32` at L=512:
    **2.5x** (fp32 default) / **4.4x** (MPP, opt-in).
  - **End-to-end** on Falcon-H1-0.5B (hybrid, so diluted): pp2048 **1.65x /
    1.85x**; decode unchanged.
  - **Perplexity neutral** on both paths over 81,920 tokens.
  - Gated to decline short sequences, where the chunked scan loses to fixed
    cost. See `integration/ggml/README.md`.
- **Milestone 5 -- production polish.** Done. The fp32 ragged tail is gone: a
  partial final chunk now reads a zero-padded copy of B and C and takes the same
  GEMM path as every other chunk, so L=96 went from **0.29x to 1.57x** and
  arbitrary sequence lengths qualify. Both metallibs are embedded in
  `libhelix.a`, so it ships with no loose files.
- **Not planned: decode.** HELIX is a prefill accelerator. A single-token step
  is bandwidth-bound, not a chunked-scan problem; ggml's kernel owns it and the
  length gate routes it there.
- **Milestone 5 -- comparative benchmarking.** Harness done; upstream baseline
  not yet captured.

## Build

```sh
xcodebuild -downloadComponent MetalToolchain    # once; Xcode 26 ships metal separately
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

The CPU reference and its tests build without the Metal toolchain; only the
kernel and its parity test need it.

## Benchmark

```sh
./build/bench_scan --L=2048 --heads=64 --head-dim=64 --d-state=128 --groups=8
```

**Read `docs/BENCHMARKING.md` before trusting any number.** GPU power state
causes a 3x spread between runs if you measure naively.

### Cumulative (M5 Max, warmed, interleaved in one process)

All at `seq=1` -- the case with no batch dimension to harvest parallelism from,
and the one that matters for interactive prefill.

| config | 1pass (fp32) | 3pass + MPP | speedup |
|---|---|---|---|
| L=2048 H=64 P=64 | 11.50 ms / 516 GFLOP/s | 1.52 ms / 3892 GFLOP/s | **7.55x** |
| L=8192 H=64 P=64 | 46.43 ms / 511 GFLOP/s | 6.23 ms / 3808 GFLOP/s | **7.45x** |

Prefill at L=8192: **176 431 -> 1 315 051 tok/s**. `spread` 1.04-1.06x.

On the M1-M4 fallback path (`HELIX_FORCE_BACKEND=sgmma`, same machine) the same
shapes give 2.47 ms and 9.84 ms -- **4.66x / 4.72x**. That is what pre-M5
hardware gets.

### How it was earned, per pass at L=8192

| pass | 1-pass baseline | + 3-pass | + `dS` GEMM | + MPP Pass C |
|---|---|---|---|---|
| A (`dS`) | | 14.85 | **1.96** | 1.96 |
| B (scan) | | 2.53 | 2.51 | 2.61 |
| C (output) | | 5.46 | 5.46 | **1.36** |
| bf16 cast | | -- | -- | 0.18 |
| **total** | 46.43 | 22.89 | 9.86 | **6.23** |

Each step was directed by this table rather than by assumption -- Pass A was 65%
of runtime and had no matmul in it at all, while Pass C (the only pass with
GEMMs) was 24%. `HELIX_ONLY_PASS=A|B|C|X` reproduces the split. Pass B, the
inter-chunk scan, is now the largest term.

### Why: occupancy

The single-pass kernel dispatches only `n_pslab x n_head x n_seq` threadgroups
and walks chunks serially inside each. At `L=8192, H=64, seq=1` that is **64
threadgroups executing 128 sequential chunk steps** on a 40-core GPU. The 3-pass
decomposition turns those 128 serial steps into two parallel waves over up to
8192 threadgroups.

An `n_seq`-scaling experiment on the single-pass kernel predicted only 1.73x
headroom (513 -> 890 GFLOP/s at 512 threadgroups). The 3-pass exceeds that at
2.04x, because batching more sequences adds memory traffic that chunk
parallelism does not.

### Scratch budget

Super-chunk tiling bounds the live scratch. It costs little:

| `HELIX_SCRATCH_MB` | groups (L=8192) | GFLOP/s |
|---|---|---|
| 64 | 4 | 1006 |
| **128** (default) | 2 | **1035** |
| 512 | 1 | 1051 |

4% between a 64 MB cap and no cap at all, so the default is comfortable.

## Layout

```
include/helix/helix.h   stable C ABI: encode-only, never allocates, never commits
src/reference/          fp64 CPU ground truth (sequential + chunked)
src/kernels/            MSL
src/runtime/            device, metallib + pipeline cache, caps, encoder
tests/                  algebra equivalence, GPU parity
bench/                  GPU-timed benchmark
```
