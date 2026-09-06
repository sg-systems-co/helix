// dispatch.mm -- the encoder.
//
// HELIX appends encoders to a command buffer the caller owns and never commits,
// waits, or allocates. That is what lets it drop into ggml's graph submission
// without adding a sync point.

#import <Metal/Metal.h>

#include <algorithm>
#include <cstdlib>

#include "helix_internal.h"

namespace {

inline id<MTLBuffer> buf_of(helix_tensor t) { return (__bridge id<MTLBuffer>)t.buf; }

constexpr NSUInteger kThreadsPerTG = HELIX_NSG * HELIX_SIMD_WIDTH;

struct Bindings {
    helix_tensor s0, x, dt, A, B, C, ids, y, s1;
};

void bind_common(id<MTLComputeCommandEncoder> enc, const Bindings& b) {
    [enc setBuffer:buf_of(b.s0) offset:b.s0.offset atIndex:1];
    [enc setBuffer:buf_of(b.x)  offset:b.x.offset  atIndex:2];
    [enc setBuffer:buf_of(b.dt) offset:b.dt.offset atIndex:3];
    [enc setBuffer:buf_of(b.A)  offset:b.A.offset  atIndex:4];
    [enc setBuffer:buf_of(b.B)  offset:b.B.offset  atIndex:5];
    [enc setBuffer:buf_of(b.C)  offset:b.C.offset  atIndex:6];
    // Metal requires a bound buffer even when the kernel never reads it; the
    // has_ids flag is what actually gates the access.
    [enc setBuffer:(b.ids.buf ? buf_of(b.ids) : buf_of(b.s0))
            offset:(b.ids.buf ? b.ids.offset : 0) atIndex:7];
    [enc setBuffer:buf_of(b.y)  offset:b.y.offset  atIndex:8];
    [enc setBuffer:buf_of(b.s1) offset:b.s1.offset atIndex:9];
}

}  // namespace

namespace {

helix_status encode_core(helix_ctx*                   ctx,
                         id<MTLComputeCommandEncoder> enc,
                         const helix_scan_desc*       d,
                         helix_tensor s0, helix_tensor x,
                         helix_tensor dt, helix_tensor A,
                         helix_tensor B,  helix_tensor C,
                         helix_tensor ids,
                         helix_tensor y,  helix_tensor s1,
                         helix_tensor scratch);

}  // namespace

extern "C" helix_status helix_scan_encode(helix_ctx*             ctx,
                                          void*                  cmd_buffer,
                                          const helix_scan_desc* d,
                                          helix_tensor s0, helix_tensor x,
                                          helix_tensor dt, helix_tensor A,
                                          helix_tensor B,  helix_tensor C,
                                          helix_tensor ids,
                                          helix_tensor y,  helix_tensor s1,
                                          helix_tensor scratch) {
    if (!ctx || !cmd_buffer || !d) return HELIX_ERR_INVALID_ARG;

    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)cmd_buffer;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    enc.label = @"helix_scan";

    const helix_status st = encode_core(ctx, enc, d, s0, x, dt, A, B, C, ids, y, s1, scratch);

    [enc endEncoding];
    return st;
}

extern "C" helix_status helix_scan_encode_into(helix_ctx*             ctx,
                                               void*                  encoder,
                                               const helix_scan_desc* d,
                                               helix_tensor s0, helix_tensor x,
                                               helix_tensor dt, helix_tensor A,
                                               helix_tensor B,  helix_tensor C,
                                               helix_tensor ids,
                                               helix_tensor y,  helix_tensor s1,
                                               helix_tensor scratch) {
    if (!ctx || !encoder || !d) return HELIX_ERR_INVALID_ARG;
    return encode_core(ctx, (__bridge id<MTLComputeCommandEncoder>)encoder,
                       d, s0, x, dt, A, B, C, ids, y, s1, scratch);
}

namespace {

helix_status encode_core(helix_ctx*                   ctx,
                         id<MTLComputeCommandEncoder> enc,
                         const helix_scan_desc*       d,
                         helix_tensor s0, helix_tensor x,
                         helix_tensor dt, helix_tensor A,
                         helix_tensor B,  helix_tensor C,
                         helix_tensor ids,
                         helix_tensor y,  helix_tensor s1,
                         helix_tensor scratch) {
    if (!s0.buf || !x.buf || !dt.buf || !A.buf || !B.buf || !C.buf || !y.buf || !s1.buf)
        return HELIX_ERR_INVALID_ARG;

    if (!helix_scan_supported(ctx, d)) return HELIX_ERR_UNSUPPORTED_SHAPE;

    HelixScanParams params{};
    if (!helix::rt::build_params(ctx, *d, &params)) return HELIX_ERR_INVALID_ARG;
    params.has_ids = ids.buf ? 1 : 0;

    if (!ctx->has_simdgroup_mm) return HELIX_ERR_UNSUPPORTED_SHAPE;

    // Resolved once. select_traits() never fails -- it degrades to the portable
    // simdgroup_matrix kernels whenever the MPP path is unavailable or the shape
    // falls outside its fixed tile extents.
    const HelixMatmulTraits& tr = helix::rt::select_traits(ctx, *d);

    const Bindings bind{s0, x, dt, A, B, C, ids, y, s1};

    // ------------------------------------------------------------------
    // Single pass
    // ------------------------------------------------------------------
    if (!d->multipass) {
        id<MTLComputePipelineState> pso = helix::rt::pipeline(ctx, tr.k_single);
        if (!pso) return HELIX_ERR_PIPELINE;
        if (kThreadsPerTG > pso.maxTotalThreadsPerThreadgroup) {
            ctx->last_error = "threadgroup size exceeds pipeline maximum";
            return HELIX_ERR_PIPELINE;
        }

        [enc setComputePipelineState:pso];
        [enc setBytes:&params length:sizeof(params) atIndex:0];
        bind_common(enc, bind);
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)params.n_pslab,
                                              (NSUInteger)d->n_head,
                                              (NSUInteger)d->n_seq)
            threadsPerThreadgroup:MTLSizeMake(kThreadsPerTG, 1, 1)];
        return HELIX_OK;
    }

    // ------------------------------------------------------------------
    // Three passes
    // ------------------------------------------------------------------
    const helix::rt::MultipassPlan plan = helix::rt::plan_multipass(*d, tr.needs_bf16_staging);
    if (plan.n_chunks <= 0) return HELIX_ERR_INVALID_ARG;

    if (!scratch.buf) {
        ctx->last_error = "multipass requires a scratch buffer";
        return HELIX_ERR_SCRATCH_TOO_SMALL;
    }
    if (buf_of(scratch).length < scratch.offset + plan.total_bytes) {
        ctx->last_error = "scratch buffer smaller than helix_scan_scratch_size()";
        return HELIX_ERR_SCRATCH_TOO_SMALL;
    }

    id<MTLComputePipelineState> pso_state = helix::rt::pipeline(ctx, tr.k_chunk_state);
    id<MTLComputePipelineState> pso_scan  = helix::rt::pipeline(ctx, tr.k_state_scan);
    id<MTLComputePipelineState> pso_out   = helix::rt::pipeline(ctx, tr.k_chunk_out);
    if (!pso_state || !pso_scan || !pso_out) return HELIX_ERR_PIPELINE;

    id<MTLComputePipelineState> pso_cast = nil;
    if (tr.needs_bf16_staging) {
        pso_cast = helix::rt::pipeline(ctx, "helix_cast_bc_bf16");
        if (!pso_cast) return HELIX_ERR_PIPELINE;
    }

    // Only needed when the sequence does not divide evenly into chunks.
    id<MTLComputePipelineState> pso_pad = nil;
    if (params.tail_t0 >= 0) {
        pso_pad = helix::rt::pipeline(ctx, "helix_pad_tail_f32");
        if (!pso_pad) return HELIX_ERR_PIPELINE;
    }

    // Scratch strides, matching the layouts documented in helix_params.h.
    params.sc_cg  = d->n_head * d->head_dim * d->d_state;
    params.sc_seq = plan.chunks_per_group * params.sc_cg;
    params.ld_cg  = d->n_head;
    params.ld_seq = plan.chunks_per_group * params.ld_cg;

    // A serial encoder (the standalone path) already orders consecutive
    // dispatches with implicit barriers, so A -> B -> C needs nothing extra. A
    // concurrent encoder -- which is what ggml hands over when
    // use_concurrency is on -- does not, and the passes are strictly dependent.
    // Read the mode off the encoder rather than making the caller declare it.
    const bool concurrent = (enc.dispatchType == MTLDispatchTypeConcurrent);
    auto barrier = [&] {
        if (concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };

    bind_common(enc, bind);
    [enc setBuffer:buf_of(scratch) offset:scratch.offset               atIndex:10];
    [enc setBuffer:buf_of(scratch) offset:scratch.offset + plan.off_ld atIndex:11];
    [enc setBuffer:buf_of(scratch) offset:scratch.offset + plan.off_btail atIndex:15];
    [enc setBuffer:buf_of(scratch) offset:scratch.offset + plan.off_ctail atIndex:16];
    if (tr.needs_bf16_staging) {
        [enc setBuffer:buf_of(scratch) offset:scratch.offset + plan.off_bbf atIndex:12];
        [enc setBuffer:buf_of(scratch) offset:scratch.offset + plan.off_cbf atIndex:13];
        [enc setBuffer:buf_of(scratch) offset:scratch.offset + plan.off_sbf atIndex:14];
    } else {
        // Pass B writes the bf16 state copy only when the pointer is non-null,
        // so on the portable path bind 10 here purely to satisfy Metal.
        [enc setBuffer:nil offset:0 atIndex:14];
    }

    const NSUInteger grid_z = (NSUInteger)d->n_seq * (NSUInteger)params.n_pslab;
    const MTLSize    tg     = MTLSizeMake(kThreadsPerTG, 1, 1);

    // Debug-only: HELIX_ONLY_PASS=A|B|C encodes just that pass, so a benchmark
    // can attribute time per pass. The output is numerically meaningless when
    // set -- it exists purely so optimization effort lands on whichever pass
    // actually dominates.
    const char* only = std::getenv("HELIX_ONLY_PASS");
    const bool do_a    = !only || *only == 'A';
    const bool do_b    = !only || *only == 'B';
    const bool do_c    = !only || *only == 'C';
    // 'X' isolates the bf16 staging cast on its own.
    const bool do_cast = !only || *only == 'X';

    // Pad the ragged final chunk once, ahead of the group loop. Cheap (one
    // chunk of B and C per sequence) and it removes the scalar tail branch that
    // used to make L=96 slower than L=128.
    if (pso_pad && do_cast) {
        [enc setComputePipelineState:pso_pad];
        [enc setBytes:&params length:sizeof(params) atIndex:0];
        const NSUInteger w = std::min<NSUInteger>(pso_pad.threadExecutionWidth,
                                                  (NSUInteger)d->d_state);
        [enc dispatchThreads:MTLSizeMake((NSUInteger)d->d_state, HELIX_CS,
                                         (NSUInteger)d->n_seq)
              threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
        barrier();
    }

    // B and C are converted to dense bf16 once for the whole sequence, ahead of
    // the group loop: they do not change between groups, and MPP cannot reach
    // the Neural Accelerators with an f32 operand.
    if (tr.needs_bf16_staging && do_cast) {
        [enc setComputePipelineState:pso_cast];
        [enc setBytes:&params length:sizeof(params) atIndex:0];
        const NSUInteger w = std::min<NSUInteger>(pso_cast.threadExecutionWidth,
                                                  (NSUInteger)d->d_state);
        [enc dispatchThreads:MTLSizeMake((NSUInteger)d->d_state,
                                         (NSUInteger)params.n_tok_pad,
                                         (NSUInteger)d->n_seq)
              threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
        barrier();
    }

    for (int base = 0; base < plan.n_chunks; base += plan.chunks_per_group) {
        params.chunk_base  = base;
        params.n_cg        = std::min(plan.chunks_per_group, plan.n_chunks - base);
        params.first_group = (base == 0) ? 1 : 0;

        [enc setBytes:&params length:sizeof(params) atIndex:0];

        // Pass A: every chunk's own state contribution, fully independent.
        if (do_a) {
            [enc setComputePipelineState:pso_state];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)params.n_cg,
                                                  (NSUInteger)d->n_head, grid_z)
                threadsPerThreadgroup:tg];
        }

        barrier();

        // Pass B: scan the (decay, dS) pairs into per-chunk initial states.
        if (do_b) {
            [enc setComputePipelineState:pso_scan];
            [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)d->n_head, grid_z)
                threadsPerThreadgroup:tg];
        }

        barrier();

        // Pass C: every chunk's output, fully independent.
        if (do_c) {
            [enc setComputePipelineState:pso_out];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)params.n_cg,
                                                  (NSUInteger)d->n_head, grid_z)
                threadsPerThreadgroup:tg];
        }

        // Groups are sequential: the next group's Pass B consumes the state this
        // one just wrote to s1.
        barrier();
    }

    return HELIX_OK;
}

}  // namespace
