# HELIX <-> ggml-metal integration

Intercepts `GGML_OP_SSM_SCAN` in llama.cpp's Metal backend.

## Upstream diff: one `#ifdef`

Pinned at `PINNED_SHA`. The entire change to llama.cpp is 15 lines in
`ggml/src/ggml-metal/ggml-metal-ops.cpp` (plus a guarded include), inside
`ggml_metal_op_ssm_scan`:

```c
#ifdef GGML_METAL_USE_HELIX
    if (ggml_metal_op_ssm_scan_helix(ctx->enc, op)) {
        ggml_metal_op_concurrency_reset(ctx);
        return 1;
    }
#endif
```

`ggml_metal_op_ssm_scan_helix` returns false without encoding anything whenever
HELIX cannot serve the op, and control falls through to the stock dispatch.

Plus a `GGML_METAL_HELIX` block in `ggml/src/ggml-metal/CMakeLists.txt`.

## Build

```sh
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_HELIX=ON \
  -DHELIX_ROOT=$PWD -DHELIX_BUILD=$PWD/build
cmake --build third_party/llama.cpp/build -j
```

HELIX itself must be built first (`libhelix.a` plus both metallibs).

## Runtime switches

| variable | default | effect |
|---|---|---|
| `GGML_HELIX_DISABLE=1` | off | bypass HELIX entirely; stock ggml kernels |
| `GGML_HELIX_MPP=1` | off | opt into the bf16 MPP path (see caveat below) |
| `GGML_HELIX_MIN_TOKENS` | 64 | sequence length below which HELIX declines |

## Why fp32 is the default

The MPP path is faster -- 4.4x against upstream at L=512, versus 2.5x for fp32
-- but its bf16 operands cannot meet ggml's own tolerance for this op:

```
test-backend-ops -o SSM_SCAN -b MTL0
  HELIX fp32 (default)   13/13 OK
  HELIX MPP              10/13 FAIL   (ERR ~1e-5 against a 2e-7 threshold)
  stock ggml (control)   13/13 OK
```

ggml's 2e-7 bar is calibrated for an fp32 kernel; bf16 carries an 8-bit
mantissa and cannot reach it. Shipping a default that fails the host project's
own op test is not defensible regardless of speed, so MPP is opt-in for callers
who have checked the end-to-end numerics for their own model.

## The length gate

HELIX is a **prefill accelerator**. Decode (L=1) stays on ggml's kernel by
design: a single-token step is a bandwidth problem, not a chunked-scan problem,
and the 3-pass decomposition's four dispatches cannot be repaid over one token.

Measured with `test-backend-ops perf -o SSM_SCAN -b MTL0` on an M5 Max
(head_dim=64, n_head=48), gate removed so the raw crossover is visible:

| L | upstream | helix fp32 | helix mpp |
|---|---|---|---|
| 1 | 15.2us | 119.3us (0.13x) | 69.8us (0.22x) |
| 16 | 57.3us | 121.7us (0.47x) | 72.4us (0.79x) |
| 32 | 106.5us | 125.6us (0.85x) | 74.8us (1.42x) |
| 48 | 147.2us | 129.3us (1.14x) | 78.2us (1.88x) |
| 64 | 151.0us | 130.5us (1.16x) | 78.8us (1.92x) |
| 96 | 246.3us | 158.2us (1.56x) | 102.5us (2.40x) |
| 512 | 1290.0us | 502.7us (2.57x) | 277.8us (4.64x) |

HELIX costs a flat ~120us below one chunk -- that is the fixed dispatch cost,
and it no longer grows with the tail. fp32 crosses 1.0x between L=32 and L=48,
MPP between L=16 and L=24. The default sits at **one full chunk (64)**, which
clears both with margin rather than balancing on the crossover.

### Shipping defaults, verified

| L | upstream | helix fp32 | helix mpp | ragged |
|---|---|---|---|---|
| 1 | 15.3us | 1.00x | 1.00x | yes |
| 32 | 100.8us | 1.01x | 1.01x | yes |
| 64 | 150.2us | 1.15x | 1.91x | - |
| 96 | 246.7us | 1.57x | 2.42x | yes |
| 129 | 316.5us | 1.76x | 2.66x | yes |
| 200 | 492.7us | 1.75x | 3.27x | yes |
| 300 | 769.1us | 2.20x | 3.62x | yes |
| 500 | 1288.3us | 2.55x | 4.59x | yes |
| 512 | 1294.1us | 2.57x | 4.62x | - |

**Worst case 1.00x on both paths.** Ragged lengths now win as much as aligned
ones -- L=96 went from 0.29x to 1.57x -- so the old `n_tok % 64 == 0` condition
is gone and arbitrary prefill lengths qualify.

## Notes on the bridge

- **HELIX appends to ggml's open encoder.** Metal forbids a second open encoder
  on one command buffer, so the bridge uses `helix_scan_encode_into()`. HELIX
  reads the encoder's `dispatchType` and inserts barriers itself when ggml is
  running concurrent dispatch.
- **Scratch is thread-local.** llama.cpp encodes graph segments in parallel
  (`dispatch_apply(n_cb, ...)` in `ggml-metal-context.m`), each thread with its
  own command buffer, so a shared scratch buffer would be a data race. HELIX
  core still never allocates; this is the host adapter's business.
- **`dst` carries two outputs.** `y` occupies `nelements(src1)` floats and the
  state follows at `s_off`. The bridge passes two tensor views into it.
- **`K > 1` is declined.** That asks the op to retain K rollback snapshots, a
  different output contract than HELIX's single final state.
- The bridge reads `id<MTLComputeCommandEncoder>` out of ggml's opaque
  `struct ggml_metal_encoder`, whose sole member it is. That is what keeps the
  upstream diff to one `#ifdef` instead of also adding an accessor; a
  `respondsToSelector:` check turns a layout change into a clean fallback
  rather than a crash.

## End-to-end results

`tiiuae/Falcon-H1-0.5B-Instruct` Q8_0 -- a hybrid Mamba-2 + attention model, so
only some layers hit SSM_SCAN at all and everything below is diluted by the
attention, FFN and embedding work.

### Throughput (`llama-bench`, M5 Max, 3 reps)

| | upstream | HELIX fp32 | HELIX MPP |
|---|---|---|---|
| pp512 | 7478 t/s | 12523 (**1.67x**) | 14201 (**1.90x**) |
| pp2048 | 7121 t/s | 11652 (**1.64x**) | 13088 (**1.84x**) |
| tg64 | 223.1 t/s | 222.3 (1.00x) | 222.1 (1.00x) |

Decode is untouched, which is the length gate working: HELIX declines and ggml's
own kernel runs. No regression, and no benefit -- decode needs the fused
single-step kernel, not this one.

### Perplexity (`llama-perplexity`, wikitext-2 test, 40 chunks @ 2048 = 81,920 tokens)

| | PPL |
|---|---|
| upstream | 13.6028 +/- 0.22459 |
| HELIX fp32 | 13.6026 +/- 0.22459 |
| HELIX MPP (bf16) | 13.6030 +/- 0.22460 |

Both HELIX paths are numerically neutral. Note what that means for the bf16
path: it **fails ggml's op-level test at 2e-7 yet is indistinguishable end to
end**, with a delta roughly a thousandth of the standard error. The op tolerance
is far stricter than model quality requires. That is an argument for making MPP
the default eventually, but it is upstream's call to make about their own test,
not something to decide by loosening a threshold locally -- so the default here
stays fp32.

## Distribution

Both metallibs are compiled into `libhelix.a` as byte arrays
(`cmake/HelixEmbed.cmake`, the equivalent of ggml's
`GGML_METAL_EMBED_LIBRARY`), so the library ships on its own with no loose
files. Verified by copying `libhelix.a` alone into an empty directory, deleting
every `.metallib`, and linking a program against it: the context comes up and
selects the MPP backend.

Load order is explicit path -> `$HELIX_METALLIB` -> embedded bytes -> a
`.metallib` beside the executable. The compile-time build-tree path is only
consulted when nothing was embedded -- otherwise a shipped binary could silently
depend on a build directory that happens to still exist, and the embedded copy
would never be exercised.

`-DHELIX_EMBED_METALLIB=OFF` reverts to on-disk loading, which is convenient
when iterating on shaders.
