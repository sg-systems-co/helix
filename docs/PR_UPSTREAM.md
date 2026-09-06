# metal: chunk-parallel SSM_SCAN for prefill (HELIX)

**4.62x** on `SSM_SCAN` at L=512 in isolation, **1.84-1.90x** end-to-end
prefill on Falcon-H1-0.5B. Decode is untouched by design.

Based on `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`.

---

## What this is

`kernel_ssm_scan_ssd_mma_f32` runs one threadgroup per (head, sequence) and
walks chunks serially inside it. At L=8192 with 64 heads that is 64 threadgroups
executing 128 sequential chunk steps, which leaves a 40-core GPU badly underfed.

HELIX keeps the same SSD algebra and changes the decomposition:

| pass | what it does |
|---|---|
| A `helix_chunk_state` | every chunk computes its own `dS` independently |
| B `helix_state_scan` | associative scan over `(decay, dS)` -> per-chunk initial states |
| C `helix_chunk_out` | every chunk computes its output from its initial state |

A and C are embarrassingly parallel across chunks; B is ~1% of the work. Scratch
is bounded by walking super-chunk groups serially, carrying state through `s1`.

Three further changes, each measured rather than assumed:

- The chunk-state GEMM (`dS = B^T . dtX`, with the per-timestep decay folded
  into `dtX`) moves onto the matrix unit. Upstream keeps that reduction in
  token order out of a concern that reassociating it "compounds rounding
  differences at every chunk boundary" -- see the numerics section below for
  what that actually costs.
- The cumulative decay uses a parallel shuffle scan instead of the serial
  `if (tiitg == 0)` loop.
- A ragged final chunk reads a zero-padded copy of B and C, so there is one GEMM
  path for every chunk and no scalar tail.

## Results

M5 Max, `test-backend-ops perf -o SSM_SCAN -b MTL0`, head_dim=64, n_head=48:

| L | upstream | HELIX (default) | HELIX (`GGML_HELIX_MPP=1`) |
|---|---|---|---|
| 1 | 15.3us | 1.00x | 1.00x |
| 32 | 100.8us | 1.01x | 1.01x |
| 64 | 150.2us | 1.15x | 1.91x |
| 96 | 246.7us | 1.57x | 2.42x |
| 129 | 316.5us | 1.76x | 2.66x |
| 200 | 492.7us | 1.75x | 3.27x |
| 300 | 769.1us | 2.20x | 3.62x |
| 500 | 1288.3us | 2.55x | 4.59x |
| 512 | 1294.1us | **2.57x** | **4.62x** |

Worst case is 1.00x -- below the gate HELIX declines and the stock kernel runs.

`head_dim=80` (Nemotron-9B shape), where upstream's `d_inner == 64` condition
sends the op to the scalar kernel entirely: **3.02x**.

End-to-end, `tiiuae/Falcon-H1-0.5B-Instruct` Q8_0 -- a hybrid, so only some
layers reach SSM_SCAN and everything here is diluted by attention/FFN work:

| | upstream | HELIX default | HELIX MPP |
|---|---|---|---|
| pp512 | 7478 t/s | 12523 (**1.67x**) | 14201 (**1.90x**) |
| pp2048 | 7121 t/s | 11652 (**1.64x**) | 13088 (**1.84x**) |
| tg64 | 223.1 t/s | 222.3 (1.00x) | 222.1 (1.00x) |

## Architectural boundaries

These are the three things worth a maintainer's attention.

### 1. Prefill only -- hard gate at `n_seq_tokens >= 64`

HELIX is a prefill accelerator. A single-token step is bandwidth-bound, not a
chunked-scan problem, and the 3-pass decomposition's four dispatches cannot be
repaid over one token. Ungated, HELIX costs a flat ~120us below one chunk:

| L | upstream | HELIX default | HELIX MPP |
|---|---|---|---|
| 1 | 15.2us | 119.3us (0.13x) | 69.8us (0.22x) |
| 16 | 57.3us | 121.7us (0.47x) | 72.4us (0.79x) |
| 32 | 106.5us | 125.6us (0.85x) | 74.8us (1.42x) |
| 48 | 147.2us | 129.3us (1.14x) | 78.2us (1.88x) |
| 64 | 151.0us | 130.5us (1.16x) | 78.8us (1.92x) |

The default path crosses 1.0x between L=32 and L=48, MPP between L=16 and L=24.
The gate sits at one full chunk (64), clearing both with margin rather than
balancing on the crossover. Decode keeps `kernel_ssm_scan_f32`, unchanged.

`K > 1` is also declined: retaining K rollback snapshots is a different output
contract than HELIX's single final state.

### 2. Default path is fp32, to respect the 2e-7 CI tolerance

The default uses `simdgroup_matrix` at fp32 and passes
`test-backend-ops -o SSM_SCAN` **13/13**, same as the stock kernel.

That is deliberate and it costs throughput. On M5 the fastest path is bf16
through MetalPerformancePrimitives, but bf16 operands carry an 8-bit mantissa
and measure ERR ~1e-5 against this op's 2e-7 threshold, failing several cases.
Shipping a default that fails your own CI would not be defensible regardless of
speed, so it is off unless asked for.

Worth knowing when weighing that tolerance: over 81,920 tokens of wikitext-2 the
bf16 path is **numerically neutral**.

| | PPL |
|---|---|
| upstream | 13.6028 +/- 0.22459 |
| HELIX fp32 | 13.6026 |
| HELIX MPP (bf16) | **13.6030** (+0.0002) |

The delta is roughly a thousandth of the standard error. The op-level bar is far
stricter than model quality requires here -- but that is your call about your own
test, not something to work around locally.

On the fp32 default path specifically: moving the chunk-state GEMM onto the
matrix unit does **not** compound across chunk boundaries. Measured against an
fp64 reference at every 8th boundary over 8192 tokens (128 chunks), max state
drift *falls* from 7.7e-8 to 4.8e-8 -- the decay forgets old error faster than
reassociation introduces it.

### 3. M5 Neural Accelerators are opt-in (`GGML_HELIX_MPP=1`)

Measured on M5 Max, 8x8x8 tiles, independent accumulator chains:

| path | operands | TFLOP/s |
|---|---|---|
| `simdgroup_matrix` | f32 | 16.12 |
| `simdgroup_matrix` | bf16 | 10.88 |
| plain vector `fma` | f32 | 14.18 |
| MPP `matmul2d` | f32 | 15.69 |
| MPP `matmul2d` | **bf16** | **65.68** |

Two results that may be useful to the project independently of this PR:

- **`simdgroup_matrix` is not the matrix hardware on M5.** At fp32 it barely
  clears plain vector FMA (16.12 vs 14.18). Reduced precision through it is a
  *pessimization* -- 0.67x -- because it pays an operand conversion instead of
  earning the 2x fp16 vector rate the hardware does have.
- **The Neural Accelerators are reachable only via MPP, and only at bf16.**
  `bf16 x f32` measures 15.65, indistinguishable from f32, so there is no
  partial migration. Both operands must be bf16.

MPP kernels live in a separate metallib built at `-std=metal4.0` and are opened
only after an `MTLGPUFamilyApple10` check, so pre-M5 devices never load them.

## Build integration

Off by default. `-DGGML_METAL_HELIX=ON` plus `HELIX_ROOT`/`HELIX_BUILD`.

The diff to this repository is **42 lines across 2 files**:

- `ggml/src/ggml-metal/ggml-metal-ops.cpp` -- one guarded include and one
  `#ifdef` block inside `ggml_metal_op_ssm_scan`:

```c
#ifdef GGML_METAL_USE_HELIX
    if (ggml_metal_op_ssm_scan_helix(ctx->enc, op)) {
        ggml_metal_op_concurrency_reset(ctx);
        return 1;
    }
#endif
```

- `ggml/src/ggml-metal/CMakeLists.txt` -- a `GGML_METAL_HELIX` block.

`ggml_metal_op_ssm_scan_helix` encodes nothing when it returns false, so the
fallback is a plain fall-through to the existing dispatch.

**No build-system contortions.** Both metallibs are compiled to byte arrays and
linked into `libhelix.a`, the same idea as `GGML_METAL_EMBED_LIBRARY` but at
the metallib level since HELIX ships pre-compiled AIR. Verified by copying
`libhelix.a` alone into an empty directory, deleting every `.metallib`, and
linking against it. No generated sources land in this tree and no new
dependencies are added.

## Implementation notes

- HELIX appends to the encoder ggml already has open (Metal forbids a second one
  on the same command buffer) and reads its `dispatchType` to insert barriers
  itself when `use_concurrency` is on.
- Scratch is thread-local because graph segments are encoded in parallel
  (`dispatch_apply(n_cb, ...)`), each thread with its own command buffer. HELIX
  core never allocates; this is the host adapter's business.
- `dst` carries both outputs; the bridge passes `y` and the state at
  `s_off` as two views.

## Testing

| | result |
|---|---|
| `test-backend-ops -o SSM_SCAN -b MTL0` (default) | 13/13 |
| `test-backend-ops -o SSM_SCAN -b MTL0` (`GGML_HELIX_MPP=1`) | 8/13, see above |
| HELIX parity vs fp64 reference, 216 cases | pass |
| HELIX 8192-token drift, both memory regimes | pass |
| wikitext-2 perplexity, 40 chunks | neutral on both paths |

HELIX's own suite covers six dispatch modes and two decay regimes. The second
regime matters: with aggressive decay the per-chunk factor is ~1e-14, the
inter-chunk carry becomes numerically irrelevant, and a test built on it cannot
detect a bug in that carry at all -- verified by injecting a 1.002x error and
watching every number stay identical.

## Known limitations

- **Decode is unimproved.** By design; it needs a fused single-step kernel.
- **Apple-only.** The decomposition is portable, this implementation is not.
- **MPP path fails the op test** at its current tolerance. Opt-in only.
- Requires `n_head % n_group == 0`, `d_state % 8 == 0`, `head_dim % 8 == 0`,
  scalar-per-head `A`, contiguous innermost dimensions. Everything else falls
  through.
