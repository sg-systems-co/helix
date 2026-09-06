# Precision on Apple Silicon: measured, not assumed

The plan assumed "fp16/bf16 MMA is ~2x fp32". On M5 that is false, and acting on
it would have made the kernel slower. These are measured on an Apple M5 Max.

## simdgroup_matrix is not the matrix hardware

Peak throughput, 8x8x8 tiles, 8 independent accumulator chains so the probe is
issue-rate bound rather than latency bound (`bench/mma_probe`):

| path | operands | accumulate | TFLOP/s | vs fp32 |
|---|---|---|---|---|
| `simdgroup_matrix` | f32 | f32 | 16.12 | 1.00x |
| `simdgroup_matrix` | f16 | f32 | 10.88 | **0.67x** |
| `simdgroup_matrix` | bf16 | f32 | 10.88 | **0.67x** |
| `simdgroup_matrix` | bf16 | bf16 | 10.04 | 0.62x |
| plain vector `fma` | f32 | f32 | 14.18 | 0.88x |
| plain vector `fma` | f16 | f16 | 26.97 | 1.67x |

Two things fall out:

1. **`simdgroup_matrix` at fp32 (16.12) barely clears plain vector FMA (14.18).**
   It is running on the ordinary ALU pipeline, not on dedicated matrix hardware.
2. **Reduced precision through `simdgroup_matrix` is a pessimization.** The
   hardware genuinely has 2x fp16 vector throughput (26.97 vs 14.18), but the
   `simdgroup_matrix` path does not get it -- it pays a conversion per operand
   instead. Porting a GEMM from fp32 to bf16 `simdgroup_matrix` makes it **1.5x
   slower**.

## The Neural Accelerators are reachable only through MPP

`mpp::tensor_ops::matmul2d`, 64x64x64 per threadgroup, accumulator held in a
`cooperative_tensor` so the loop stays compute bound (`bench/mpp_probe`):

| path | operands | TFLOP/s | vs simdgroup fp32 |
|---|---|---|---|
| MPP `matmul2d` | f32 | 15.80 | 0.98x |
| MPP `matmul2d` | f16 | **65.81** | **4.08x** |
| MPP `matmul2d` | bf16 | **65.80** | **4.08x** |

Verified as real work: time scales exactly linearly with the repeat count
(4.18 / 8.36 / 16.71 / 33.41 ms for 128 / 256 / 512 / 1024 reps).

**fp32 through MPP buys nothing** (15.80 vs 16.12) -- it falls back to the same
vector path. **bf16 through MPP is 4.1x**, and 6.0x faster than bf16 through
`simdgroup_matrix`.

## Two further constraints, measured

Both discovered while designing the port, both load-bearing:

| variant | TFLOP/s |
|---|---|
| MPP `matmul2d`, bf16 x bf16, device operands | 65.68 |
| MPP `matmul2d`, **bf16 x f32** | **15.65** |
| MPP `matmul2d`, f32 x bf16 | 15.91 |
| MPP `matmul2d`, bf16 x bf16, **threadgroup** operands | **64.26** |

1. **Both operands must be bf16.** A mixed bf16 x f32 matmul is
   indistinguishable from plain f32, so there is no partial migration: B, C and
   the per-chunk initial state all need dense bf16 copies before Pass C can use
   the Neural Accelerators. That is what `helix_cast_bc_bf16` and the bf16
   `Sbf` region in scratch exist for.
2. **Threadgroup operands cost ~2%.** So the masked `M` tile and `dtX` stay
   on-chip and never round-trip to device memory between G1 and G2.

## What this means for HELIX

Reduced precision and the MPP path are not separable on M5:

- bf16 is only worth having **through MPP**. Through `simdgroup_matrix` it costs
  1.5x.
- MPP is only worth having **at bf16**. At fp32 it is a wash.

So the milestone that matters is "bf16 via MPP tensor ops", not "bf16" and "MPP"
as two independent steps.

This is what `HelixMatmulTraits` encodes (`src/runtime/caps.mm`):

- **`MppTraits`** -- Apple10 (M5). Pass C runs `helix_chunk_out_mpp` with bf16
  operands and fp32 accumulators. Requires `d_state == 128` and
  `head_dim % 64 == 0`, because the matmul descriptors are compile-time.
- **`SgmmaTraits`** -- Apple7-9 (M1-M4, A14+), and any shape outside the MPP
  kernel's fixed extents. fp32 `simdgroup_matrix` throughout.

Selection never fails: anything the MPP path cannot serve falls back silently.
`HELIX_FORCE_BACKEND=sgmma` pins the fallback so it can be regression-tested on
M5 hardware.

### Where precision actually lands

Only the GEMM *operands* in Pass C are narrowed. The cumulative decay, every
accumulator, `dS`, and the inter-chunk carry all stay fp32 -- so the carried
state is held to the **fp32** tolerance even on the MPP path, and the drift test
doubles as a check that bf16 staging has not leaked into the carry.

Measured tolerances, swept to the failure edge the same way as the fp32 pair:

| path | failure edge | tolerance used |
|---|---|---|
| fp32 | rel 2.6e-6 / abs 2.6e-7 | rel 5e-6 / abs 5e-7 |
| bf16 (Pass C output) | rel 7e-3 / abs 1.4e-3 | rel 1.5e-2 / abs 3e-3 |

The bf16 edge sits where bf16's 8-bit mantissa puts it (2^-8 ~ 3.9e-3).
