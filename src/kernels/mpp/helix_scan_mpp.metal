// helix_scan_mpp.metal -- Pass C on the M5 Neural Accelerators.
//
// Measured on an M5 Max (bench/probes, docs/PRECISION.md):
//
//   simdgroup_matrix  f32        16.12 TFLOP/s
//   simdgroup_matrix  bf16       10.88          <- reduced precision is a LOSS here
//   MPP matmul2d      f32        15.69
//   MPP matmul2d      bf16       65.68          <- 4.19x, and the only route to the NAs
//   MPP matmul2d      bf16, threadgroup operands
//                                64.26          <- 98% of device-operand speed
//
// Two constraints follow, and they shape this whole file:
//
//   * BOTH operands must be bf16. bf16 x f32 measures 15.65 -- indistinguishable
//     from plain f32 -- so there is no partial migration. B, C and the chunk's
//     initial state are all staged to bf16 first.
//   * Threadgroup operands are essentially free, so M and dtX never round-trip
//     through device memory between G1 and G2.
//
// Accumulators stay fp32 throughout (matmul2d's destination element type), and
// the cumulative decay is fp32 as always. Only the GEMM *operands* are narrowed.
//
// Unlike the simdgroup_matrix kernel this computes the full 64x64 M tile rather
// than just the causal triangle: MPP's execution scope is the whole threadgroup
// working on one tile, so there is no cheap way to skip half of it. That is ~2x
// the arithmetic for G1 and G2, comfortably paid for by the 4.19x rate.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "helix_params.h"

using namespace metal;
using namespace mpp::tensor_ops;

using ext2 = dextents<int32_t, 2>;

inline float helix_softplus_mpp(float v) { return v > 20.0f ? v : log(1.0f + exp(v)); }

inline float helix_simd_prefix_inclusive_mpp(float v, ushort lane) {
    float n;
    n = simd_shuffle_up(v,  1u); if (lane >=  1u) v += n;
    n = simd_shuffle_up(v,  2u); if (lane >=  2u) v += n;
    n = simd_shuffle_up(v,  4u); if (lane >=  4u) v += n;
    n = simd_shuffle_up(v,  8u); if (lane >=  8u) v += n;
    n = simd_shuffle_up(v, 16u); if (lane >= 16u) v += n;
    return v;
}

kernel void helix_chunk_out_mpp(
        constant HelixScanParams& p   [[buffer(0)]],
        device const float*       x   [[buffer(2)]],
        device const float*       dtr [[buffer(3)]],
        device const float*       A   [[buffer(4)]],
        device       float*       y   [[buffer(8)]],
        device       bfloat*      Cbf [[buffer(13)]],
        device       bfloat*      Bbf [[buffer(12)]],
        device       bfloat*      Sbf [[buffer(14)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {

    constexpr short  CS  = HELIX_CS;
    constexpr short  HD  = HELIX_HD;
    constexpr ushort TGS = HELIX_NSG * HELIX_SIMD_WIDTH;

    // ~17 KiB: less than half the simdgroup_matrix kernel's budget, because the
    // large operands (C, B, S) are read straight from device memory as bf16.
    threadgroup bfloat Mtile[CS * CS];     //  8 KiB
    threadgroup bfloat dtX[CS * HD];       //  8 KiB
    threadgroup float  Lcum[CS];
    threadgroup float  exp_Lcum[CS];
    threadgroup float  sg_scratch[HELIX_NSG];

    const int slab = int(tgpig.z) % p.n_pslab;
    const int seq  = int(tgpig.z) / p.n_pslab;
    const int h    = int(tgpig.y);
    const int p0   = slab * HD;
    const int pw   = min(int(HD), p.head_dim - p0);

    const int cg = int(tgpig.x);
    if (cg >= p.n_cg) return;

    const int t0 = (p.chunk_base + cg) * CS;
    if (t0 >= p.n_tok) return;
    const short tlen = short(min(int(CS), p.n_tok - t0));

    const int   N   = p.d_state;
    const int   g   = h / p.heads_per_group;
    const float A_h = A[h * p.s_a];

    device const float* Xp  = x   + seq * p.seq_x  + h * p.s_x[1];
    device const float* DTp = dtr + seq * p.seq_dt + h * p.s_dt[1];
    device       float* Yp  = y   + seq * p.seq_y  + h * p.s_y[1];

    // ---------------------------------------------------------------------
    // Preamble: dtX (bf16 operand) and the fp32 cumulative decay.
    // ---------------------------------------------------------------------
    for (int idx = tiitg; idx < CS * HD; idx += TGS) {
        const int t = idx / HD;
        const int c = idx % HD;
        float v = 0.0f;
        if (t < tlen && c < pw) {
            float d = DTp[(t0 + t) * p.s_dt[0]];
            d = clamp(helix_softplus_mpp(d), p.dt_min, p.dt_max);
            v = d * Xp[(t0 + t) * p.s_x[0] + (p0 + c) * p.s_x[2]];
        }
        dtX[idx] = bfloat(v);
    }

    {
        float a = 0.0f;
        if (tiitg < tlen) {
            float d = DTp[(t0 + tiitg) * p.s_dt[0]];
            a = A_h * clamp(helix_softplus_mpp(d), p.dt_min, p.dt_max);
        }
        const float prefix = helix_simd_prefix_inclusive_mpp(a, tiisg);
        if (tiisg == HELIX_SIMD_WIDTH - 1) sg_scratch[sgitg] = prefix;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg == 0) {
            const ushort n_sg = (tlen + HELIX_SIMD_WIDTH - 1) / HELIX_SIMD_WIDTH;
            float running = 0.0f;
            for (ushort s = 0; s < n_sg; ++s) {
                const float tot = sg_scratch[s];
                sg_scratch[s] = running;
                running += tot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg < tlen) {
            Lcum[tiitg]     = prefix + sg_scratch[sgitg];
            exp_Lcum[tiitg] = exp(Lcum[tiitg]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Dense bf16 views of this chunk's C and B. Row stride is n_group*d_state
    // because the staging copy keeps every group side by side.
    const int bf_off = seq * p.bf_seq + t0 * p.bf_tok + g * N;
    const array<int32_t, 2> rc_strides = {1, p.bf_tok};

    tensor<device bfloat, ext2, tensor_inline> tC(Cbf + bf_off, ext2(N, CS), rc_strides);
    tensor<device bfloat, ext2, tensor_inline> tB(Bbf + bf_off, ext2(N, CS), rc_strides);

    // ---------------------------------------------------------------------
    // G1: M_raw = C . B^T   (CS x N)(N x CS) -> CS x CS
    // then mask and decay in place, narrowing to the bf16 operand G2 needs.
    // ---------------------------------------------------------------------
    {
        constexpr auto d1 = matmul2d_descriptor(CS, CS, HELIX_MPP_K,
                                                /*transpose_left */ false,
                                                /*transpose_right*/ true);
        matmul2d<d1, execution_simdgroups<HELIX_NSG>> op1;
        auto acc = op1.get_destination_cooperative_tensor<decltype(tC), decltype(tB), float>();
#pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) acc[i] = 0.0f;

        op1.run(tC, tB, acc);

        // Causal mask and the intra-chunk decay, applied on the cooperative
        // tensor's own elements -- G1's result never leaves registers except as
        // the bf16 tile G2 consumes.
#pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            const auto ij = acc.get_multidimensional_index(i);
            const int  t  = int(ij[1]);   // row: query timestep
            const int  j  = int(ij[0]);   // col: key timestep
            float v = 0.0f;
            if (j <= t && t < tlen && j < tlen)
                v = acc[i] * exp(Lcum[t] - Lcum[j]);
            Mtile[t * CS + j] = bfloat(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    tensor<threadgroup bfloat, ext2, tensor_inline> tM(Mtile, ext2(CS, CS));
    tensor<threadgroup bfloat, ext2, tensor_inline> tD(dtX,   ext2(HD, CS));

    // ---------------------------------------------------------------------
    // G2: Y_intra = M . dtX          (CS x CS)(CS x HD)
    // G3: Y_inter = C . S_init       (CS x N )(N  x HD)
    // ---------------------------------------------------------------------
    constexpr auto d2 = matmul2d_descriptor(CS, HD, CS);
    matmul2d<d2, execution_simdgroups<HELIX_NSG>> op2;
    auto y_intra = op2.get_destination_cooperative_tensor<decltype(tM), decltype(tD), float>();
#pragma unroll
    for (uint16_t i = 0; i < y_intra.get_capacity(); ++i)
        if (y_intra.is_valid_element(i)) y_intra[i] = 0.0f;
    op2.run(tM, tD, y_intra);

    device bfloat* s_in =
        Sbf + seq * p.sc_seq + cg * p.sc_cg + h * N * p.head_dim + p0;
    const array<int32_t, 2> s_strides = {1, p.head_dim};
    tensor<device bfloat, ext2, tensor_inline> tS(s_in, ext2(HD, N), s_strides);

    constexpr auto d3 = matmul2d_descriptor(CS, HD, HELIX_MPP_K);
    matmul2d<d3, execution_simdgroups<HELIX_NSG>> op3;
    auto y_inter = op3.get_destination_cooperative_tensor<decltype(tC), decltype(tS), float>();
#pragma unroll
    for (uint16_t i = 0; i < y_inter.get_capacity(); ++i)
        if (y_inter.is_valid_element(i)) y_inter[i] = 0.0f;
    op3.run(tC, tS, y_inter);

    // y[t][c] = y_intra + exp(Lcum_t) * y_inter
    //
    // Both cooperative tensors come from the same descriptor shape and scope, so
    // they distribute elements identically and index i means the same (t, c) in
    // each.
#pragma unroll
    for (uint16_t i = 0; i < y_intra.get_capacity(); ++i) {
        if (!y_intra.is_valid_element(i)) continue;
        const auto tc = y_intra.get_multidimensional_index(i);
        const int  t  = int(tc[1]);
        const int  c  = int(tc[0]);
        if (t >= tlen || c >= pw) continue;
        Yp[(t0 + t) * p.s_y[0] + (p0 + c) * p.s_y[2]] =
            y_intra[i] + exp_Lcum[t] * y_inter[i];
    }
}
