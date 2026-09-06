// helix_common.h -- device-side helpers shared by the HELIX kernels.

#ifndef HELIX_COMMON_H
#define HELIX_COMMON_H

#include <metal_stdlib>
#include "helix_params.h"

using namespace metal;

// The x > 20 shortcut is not just an optimization: ggml-metal uses exactly this
// form, and without it HELIX and upstream disagree in the tail on inputs where
// both are individually defensible.
inline float helix_softplus(float v) {
    return v > 20.0f ? v : log(1.0f + exp(v));
}

inline float helix_dt(constant HelixScanParams& p, float raw) {
    return clamp(helix_softplus(raw), p.dt_min, p.dt_max);
}

// Kogge-Stone inclusive prefix sum across one 32-lane simdgroup.
inline float helix_simd_prefix_inclusive(float v, ushort lane) {
    float n;
    n = simd_shuffle_up(v,  1u); if (lane >=  1u) v += n;
    n = simd_shuffle_up(v,  2u); if (lane >=  2u) v += n;
    n = simd_shuffle_up(v,  4u); if (lane >=  4u) v += n;
    n = simd_shuffle_up(v,  8u); if (lane >=  8u) v += n;
    n = simd_shuffle_up(v, 16u); if (lane >= 16u) v += n;
    return v;
}

// Inclusive prefix sum of `v` over the first CS threads of the threadgroup,
// written to `out`. `scratch` needs HELIX_NSG floats.
//
// This replaces the serial single-thread cumsum that ggml-metal runs at the top
// of every chunk (`if (tiitg == 0) { for (t...) acc += ... }`), which is the
// textbook serial bottleneck in an otherwise parallel kernel.
inline void helix_chunk_prefix_sum(float v, ushort tiitg, ushort sgitg, ushort tiisg,
                                   ushort n_active,
                                   threadgroup float* out,
                                   threadgroup float* scratch) {
    const float prefix = helix_simd_prefix_inclusive(v, tiisg);

    if (tiisg == HELIX_SIMD_WIDTH - 1) scratch[sgitg] = prefix;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One lane turns the per-simdgroup totals into an exclusive prefix. n_sg is
    // at most HELIX_NSG (4), so a serial loop here costs nothing.
    if (tiitg == 0) {
        const ushort n_sg = (n_active + HELIX_SIMD_WIDTH - 1) / HELIX_SIMD_WIDTH;
        float running = 0.0f;
        for (ushort s = 0; s < n_sg; ++s) {
            const float tot = scratch[s];
            scratch[s] = running;
            running += tot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiitg < n_active) out[tiitg] = prefix + scratch[sgitg];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------------------------------------------------------------------
// Shared chunk preamble
// ---------------------------------------------------------------------------
//
// Computes, for one chunk of one (sequence, head, head-dim slab):
//
//   dtX[j][c]   = softplus(dt_j) * x[j][c]      rows >= tlen zeroed
//   Lcum[t]     = sum_{j<=t} A_h * dt_j         parallel shuffle scan
//   exp_Lcum[t] = exp(Lcum[t])
//   sdecay[j]   = exp(Lcum[tlen-1] - Lcum[j])
//
// Both the chunk-state pass and the output pass need all of this. Recomputing
// it in each is deliberately cheaper than round-tripping dtX (CS*HD floats per
// chunk-head) through device memory.
//
// Returns the chunk-final cumulative log-decay. On return every threadgroup
// array is visible to the whole threadgroup.
inline float helix_chunk_preamble(constant HelixScanParams& p,
                                  device const float* DTp,
                                  device const float* Xp,
                                  int t0, short tlen, int p0, int pw, float A_h,
                                  ushort tiitg, ushort sgitg, ushort tiisg, ushort tgs,
                                  threadgroup float* dtX,
                                  threadgroup float* Lcum,
                                  threadgroup float* exp_Lcum,
                                  threadgroup float* sdecay,
                                  threadgroup float* sg_scratch) {
    for (int idx = tiitg; idx < HELIX_CS * HELIX_HD; idx += tgs) {
        const int t = idx / HELIX_HD;
        const int c = idx % HELIX_HD;
        float v = 0.0f;
        if (t < tlen && c < pw) {
            const float dt_t = helix_dt(p, DTp[(t0 + t) * p.s_dt[0]]);
            v = dt_t * Xp[(t0 + t) * p.s_x[0] + (p0 + c) * p.s_x[2]];
        }
        dtX[idx] = v;
    }

    {
        const float a = (tiitg < tlen)
            ? A_h * helix_dt(p, DTp[(t0 + tiitg) * p.s_dt[0]])
            : 0.0f;
        helix_chunk_prefix_sum(a, tiitg, sgitg, tiisg, ushort(tlen), Lcum, sg_scratch);
    }

    // Rows past tlen are clamped to the chunk-final value rather than left
    // undefined. The padded GEMM path evaluates exp(Lcum[i] - Lcum[j]) over the
    // full HELIX_CS tile and multiplies by a zero from the padded operand;
    // garbage here could be +inf, and 0 * inf is NaN, not zero. Clamping keeps
    // every exponent <= 0 and therefore every factor in (0, 1].
    if (tiitg < HELIX_CS) {
        const float lc = (tiitg < tlen) ? Lcum[tiitg] : Lcum[tlen - 1];
        Lcum[tiitg]     = lc;
        exp_Lcum[tiitg] = exp(lc);
        sdecay[tiitg]   = (tiitg < tlen) ? exp(Lcum[tlen - 1] - lc) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return Lcum[tlen - 1];
}

#endif  // HELIX_COMMON_H
