// ggml-helix.cpp -- the ggml-metal <-> HELIX bridge.
//
// Everything ggml-specific lives here so the upstream diff stays a single
// #ifdef. Three things make that possible:
//
//   * ggml_metal_buffer_id is {void* metal; size_t offs}, which is exactly
//     helix_tensor, so tensors map across without a copy.
//   * ggml's nb[] are byte strides and helix_scan_desc wants element strides,
//     so the mapping is a divide, not a repack.
//   * HELIX appends to ggml's already-open encoder via
//     helix_scan_encode_into(), because Metal forbids a second open encoder on
//     the same command buffer.
//
// Layouts (ggml side, fastest dimension first):
//
//   src0  s     {d_state, head_dim, n_head, n_seqs}
//   src1  x     {head_dim, n_head, n_tokens, n_seqs}
//   src2  dt    {n_head, n_tokens, n_seqs}
//   src3  A     {1 | d_state, n_head}
//   src4  B     {d_state, n_group, n_tokens, n_seqs}
//   src5  C     {d_state, n_group, n_tokens, n_seqs}
//   src6  ids   {n_seqs}                                   int32
//   dst         y, then the output state at byte offset s_off
//
// Note that dst carries BOTH outputs: y occupies nelements(src1) floats and the
// state follows it. HELIX writes them as two separate tensor views into the
// same buffer.

#include "ggml-helix.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-metal-device.h"
#include "ggml.h"
#include "helix/helix.h"

#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {

// --------------------------------------------------------------------------
// Context
// --------------------------------------------------------------------------

helix_ctx *    g_ctx      = nullptr;
bool           g_disabled = false;
bool           g_use_mpp  = false;
int            g_min_tok  = 64;
std::once_flag g_once;

helix_ctx * helix_context() {
    std::call_once(g_once, [] {
        if (const char * off = std::getenv("GGML_HELIX_DISABLE"); off && off[0] == '1') {
            g_disabled = true;
            return;
        }
        // The MPP path is opt-in, not the default.
        //
        // It is the fastest thing HELIX has (bf16 on the M5 Neural Accelerators,
        // 7.45x over the single-pass baseline against 4.7x for fp32), but bf16
        // operands carry an 8-bit mantissa and ggml's own SSM_SCAN test is
        // calibrated at 2e-7 -- an fp32 bar. HELIX/MPP measures ~1e-5 there and
        // fails 7 of 13 cases, while HELIX/fp32 passes 13/13.
        //
        // Defaulting to a path that fails the host project's own op test would
        // be indefensible regardless of how fast it is, so the default is fp32
        // and GGML_HELIX_MPP=1 opts in for anyone who has checked that the
        // end-to-end numerics hold for their model.
        if (const char * m = std::getenv("GGML_HELIX_MPP"); m && m[0] == '1') {
            g_use_mpp = true;
        }

        if (const char * t = std::getenv("GGML_HELIX_MIN_TOKENS")) {
            g_min_tok = std::atoi(t);
        }

        const helix_status st = helix_ctx_create(&g_ctx, nullptr, nullptr);
        if (st != HELIX_OK) {
            g_ctx = nullptr;
            std::fprintf(stderr, "%s: HELIX unavailable (status %d), using ggml kernels\n", __func__, (int) st);
        }
    });
    return g_disabled ? nullptr : g_ctx;
}

// --------------------------------------------------------------------------
// Scratch
// --------------------------------------------------------------------------
//
// HELIX itself never allocates -- that property is what lets it drop into a
// graph -- so the host adapter owns the scratch buffer.
//
// It is thread-local because llama.cpp encodes graph segments in parallel:
// ggml-metal-context.m runs `dispatch_apply(n_cb, ...)`, giving each thread its
// own command buffer. A single shared scratch buffer would be a genuine data
// race between those threads. Reuse across graphs is safe because
// ggml_metal_graph_compute waits on every command buffer before returning.
//
// Buffers grow monotonically and are never released; the bound is one per
// encoding thread, which is n_cb + 1.
thread_local void * t_scratch      = nullptr;
thread_local size_t t_scratch_size = 0;

id<MTLBuffer> scratch_for(id<MTLDevice> dev, size_t bytes) {
    if (bytes == 0) {
        return nil;
    }
    if (t_scratch && t_scratch_size >= bytes) {
        return (__bridge id<MTLBuffer>) t_scratch;
    }

    if (t_scratch) {
        CFRelease(t_scratch);
        t_scratch      = nullptr;
        t_scratch_size = 0;
    }

    id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    if (!buf) {
        return nil;
    }

    t_scratch      = (void *) CFBridgingRetain(buf);
    t_scratch_size = bytes;
    return buf;
}

// --------------------------------------------------------------------------
// Mapping
// --------------------------------------------------------------------------

// Mirrors the static helper of the same name in ggml-metal-ops.cpp. The
// view_src indirection matters: ggml hands over views of a larger allocation,
// and the Metal buffer lives on the source.
ggml_metal_buffer_id buffer_id(const ggml_tensor * t) {
    if (!t) {
        return { nullptr, 0 };
    }
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_metal_buffer_t   bctx   = (ggml_metal_buffer_t) buffer->context;
    return ggml_metal_buffer_get_id(bctx, t);
}

inline helix_tensor to_helix(ggml_metal_buffer_id b) {
    return helix_tensor{ b.metal, (uint64_t) b.offs };
}

// ggml nb[] are byte strides; helix_scan_desc wants element counts.
inline int64_t elems(uint64_t nb, size_t type_size) {
    return (int64_t) (nb / type_size);
}

bool build_desc(const ggml_tensor * op, helix_scan_desc * d) {
    const ggml_tensor * src0 = op->src[0];  // s
    const ggml_tensor * src1 = op->src[1];  // x
    const ggml_tensor * src2 = op->src[2];  // dt
    const ggml_tensor * src3 = op->src[3];  // A
    const ggml_tensor * src4 = op->src[4];  // B
    const ggml_tensor * src5 = op->src[5];  // C

    // HELIX produces one final state. K > 1 asks the op to retain K rollback
    // snapshots, which is a different output contract -- leave it to ggml.
    if (ggml_get_op_params_i32(op, 0) != 1) {
        return false;
    }

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_F32 ||
        src3->type != GGML_TYPE_F32 || src4->type != GGML_TYPE_F32 || src5->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t d_state  = src0->ne[0];
    const int64_t head_dim = src0->ne[1];  // ggml calls this d_inner
    const int64_t n_head   = src0->ne[2];
    const int64_t n_group  = src4->ne[1];
    const int64_t n_tok    = src1->ne[2];
    const int64_t n_seq    = src1->ne[3];

    if (n_head <= 0 || n_group <= 0 || n_head % n_group != 0) {
        return false;
    }

    // ------------------------------------------------------------------
    // Where HELIX is actually faster. Measured against the stock ggml kernel
    // on an M5 Max via test-backend-ops perf (head_dim=64, n_head=48), with the
    // gate removed so the raw crossover is visible:
    //
    //     L      upstream   helix f32   helix mpp
    //       1      15.2us    119.3us      69.8us     0.13x  0.22x
    //      16      57.3us    121.7us      72.4us     0.47x  0.79x
    //      32     106.5us    125.6us      74.8us     0.85x  1.42x
    //      48     147.2us    129.3us      78.2us     1.14x  1.88x
    //      64     151.0us    130.5us      78.8us     1.16x  1.92x
    //      96     246.3us    158.2us     102.5us     1.56x  2.40x
    //     512    1290.0us    502.7us     277.8us     2.57x  4.64x
    //
    // HELIX costs a flat ~120us below one chunk: the 3-pass decomposition is
    // four dispatches with barriers between them, and no amount of parallelism
    // repays that over a handful of tokens. fp32 crosses 1.0x between L=32 and
    // L=48; MPP between L=16 and L=24. The default sits at one full chunk,
    // which clears both with margin rather than balancing on the crossover.
    //
    // Decode (L=1) is 7x slower and always will be -- it is a bandwidth
    // problem, not a chunked-scan problem, and belongs in a fused single-step
    // kernel. HELIX is a prefill accelerator; ggml's kernel owns decode.
    //
    // The fp32 path used to collapse on a partial final chunk (L=96 was slower
    // than L=128) because its ragged branch computed y_inter one thread per
    // token with a head_dim-strided read. That branch is gone: the ragged chunk
    // now reads a zero-padded copy of B and C and takes the same GEMM path as
    // every other chunk, so arbitrary sequence lengths qualify.
    if (n_tok < g_min_tok) {
        return false;
    }

    helix_scan_desc_init(d, (int32_t) n_tok, (int32_t) n_seq, (int32_t) n_head, (int32_t) head_dim, (int32_t) d_state,
                         (int32_t) n_group);

    const size_t ts = sizeof(float);

    // s: (head, chan, state) within a sequence.
    d->s_s[0] = elems(src0->nb[2], ts);
    d->s_s[1] = elems(src0->nb[1], ts);
    d->s_s[2] = elems(src0->nb[0], ts);
    d->seq_s  = elems(src0->nb[3], ts);

    // x: (tok, head, chan).
    d->s_x[0] = elems(src1->nb[2], ts);
    d->s_x[1] = elems(src1->nb[1], ts);
    d->s_x[2] = elems(src1->nb[0], ts);
    d->seq_x  = elems(src1->nb[3], ts);

    // dt: (tok, head).
    d->s_dt[0] = elems(src2->nb[1], ts);
    d->s_dt[1] = elems(src2->nb[0], ts);
    d->seq_dt  = elems(src2->nb[2], ts);

    // B and C: (tok, group, state).
    d->s_b[0] = elems(src4->nb[2], ts);
    d->s_b[1] = elems(src4->nb[1], ts);
    d->s_b[2] = elems(src4->nb[0], ts);
    d->seq_b  = elems(src4->nb[3], ts);

    d->s_c[0] = elems(src5->nb[2], ts);
    d->s_c[1] = elems(src5->nb[1], ts);
    d->s_c[2] = elems(src5->nb[0], ts);
    d->seq_c  = elems(src5->nb[3], ts);

    // A is {1, n_head} for the scalar-decay (Mamba-2 / SSD) form and
    // {d_state, n_head} for a per-channel diagonal. ggml's own fast path gates
    // on ne[0] == 1 exactly the same way.
    if (src3->ne[0] == 1) {
        d->a_form = HELIX_A_SCALAR_PER_HEAD;
    } else if (src3->ne[0] == d_state) {
        d->a_form = HELIX_A_DIAG_PER_CHANNEL;
    } else {
        return false;
    }
    d->s_a = elems(src3->nb[1], ts);

    // y is the leading region of dst and is written densely, matching x.
    d->s_y[0] = n_head * head_dim;
    d->s_y[1] = head_dim;
    d->s_y[2] = 1;
    d->seq_y  = n_tok * n_head * head_dim;

    // ggml applies no clamp to softplus(dt); match that exactly or the outputs
    // diverge in the tail on large dt.
    d->dt_min = 0.0f;
    d->dt_max = INFINITY;

    d->multipass = true;
    d->backend   = g_use_mpp ? HELIX_BACKEND_AUTO : HELIX_BACKEND_SGMMA;
    return true;
}

}  // namespace

// --------------------------------------------------------------------------

bool ggml_metal_op_ssm_scan_helix(struct ggml_metal_encoder * enc, const ggml_tensor * op) {
    helix_ctx * ctx = helix_context();
    if (!ctx || !enc || !op) {
        return false;
    }

    helix_scan_desc d{};
    if (!build_desc(op, &d)) {
        return false;
    }
    if (!helix_scan_supported(ctx, &d)) {
        return false;
    }

    // struct ggml_metal_encoder is opaque in ggml-metal-device.h, but its
    // definition (ggml-metal-device.m) is a single id<MTLComputeCommandEncoder>
    // member. Reading it here rather than adding an accessor upstream is what
    // keeps the ggml diff to one #ifdef. The responds-to-selector check below
    // turns a layout change into a clean fallback instead of a crash.
    id<MTLComputeCommandEncoder> mtl_enc = *(id<MTLComputeCommandEncoder> __unsafe_unretained *) enc;
    if (!mtl_enc || ![mtl_enc respondsToSelector:@selector(dispatchThreadgroups:threadsPerThreadgroup:)]) {
        return false;
    }

    id<MTLDevice> dev = mtl_enc.device;

    const size_t  scratch_bytes = helix_scan_scratch_size(ctx, &d);
    id<MTLBuffer> scratch       = scratch_for(dev, scratch_bytes);
    if (scratch_bytes && !scratch) {
        return false;
    }

    const helix_tensor t_s0  = to_helix(buffer_id(op->src[0]));
    const helix_tensor t_x   = to_helix(buffer_id(op->src[1]));
    const helix_tensor t_dt  = to_helix(buffer_id(op->src[2]));
    const helix_tensor t_A   = to_helix(buffer_id(op->src[3]));
    const helix_tensor t_B   = to_helix(buffer_id(op->src[4]));
    const helix_tensor t_C   = to_helix(buffer_id(op->src[5]));
    const helix_tensor t_ids = to_helix(buffer_id(op->src[6]));

    // dst holds y first, then the output state at s_off bytes.
    const ggml_metal_buffer_id dst   = buffer_id(op);
    const size_t               s_off = ggml_nelements(op->src[1]) * sizeof(float);

    const helix_tensor t_y  = helix_tensor{ dst.metal, (uint64_t) dst.offs };
    const helix_tensor t_s1 = helix_tensor{ dst.metal, (uint64_t) (dst.offs + s_off) };

    const helix_tensor t_scratch = scratch ? helix_tensor{ (__bridge void *) scratch, 0 } : HELIX_TENSOR_NULL;

    const helix_status st = helix_scan_encode_into(ctx, (__bridge void *) mtl_enc, &d, t_s0, t_x, t_dt, t_A, t_B, t_C,
                                                   t_ids, t_y, t_s1, t_scratch);

    if (st != HELIX_OK) {
        // Nothing was dispatched on a descriptor rejection, so falling through
        // to ggml's own dispatch is safe.
        std::fprintf(stderr, "%s: helix_scan_encode_into failed (%d): %s\n", __func__, (int) st,
                     helix_ctx_last_error(ctx));
        return false;
    }

    return true;
}
