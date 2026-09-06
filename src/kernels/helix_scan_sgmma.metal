// helix_scan_sgmma.metal -- chunked associative scan (SSD) on simdgroup_matrix.
//
// Milestone 2. This deliberately mirrors ggml-metal's kernel_ssm_scan_ssd_mma_f32
// structure -- fp32 throughout, state spilled to device memory between chunks --
// so that every later change (on-chip bf16 state, MMA state update, chunk
// parallelism) can be attributed to one variable. The two things it does NOT
// mirror are both correctness, not performance:
//
//   * The cumulative decay uses a parallel shuffle scan, not a serial loop on
//     thread 0.
//   * Ragged chunks (L % CS != 0) are handled in-kernel instead of being punted
//     to a separate scalar kernel, and head_dim > 64 is covered by splitting the
//     channel dimension across threadgroups (grid.x). Those two gates are where
//     upstream falls back entirely.
//
// Grid:         (n_pslab, n_head, n_seq)
// Threadgroup:  HELIX_NSG * 32 threads
//
// Threadgroup memory (fp32):
//   dtX      CS x HD          16384 B
//   sam      NSG x 8 x CS      8192 B
//   tiles    NSG x 2 x 8 x 8   2048 B
//   Lcum / exp_Lcum / decay    3 x CS x 4 = 768 B
//   sg_scratch                   16 B
//                             ------------
//                             ~26.8 KiB, same budget as upstream.

#include "helix_common.h"

using namespace metal;

kernel void helix_scan_sgmma_f32(
        constant HelixScanParams& p   [[buffer(0)]],
        device const float*       s0  [[buffer(1)]],
        device const float*       x   [[buffer(2)]],
        device const float*       dtr [[buffer(3)]],
        device const float*       A   [[buffer(4)]],
        device const float*       Bt  [[buffer(5)]],
        device const float*       Ct  [[buffer(6)]],
        device const int*         ids [[buffer(7)]],
        device       float*       y   [[buffer(8)]],
        device       float*       s1  [[buffer(9)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {

    constexpr short CS  = HELIX_CS;
    constexpr short HD  = HELIX_HD;
    constexpr short TC  = HELIX_TC;
    constexpr short NSG = HELIX_NSG;
    constexpr short TGS = NSG * HELIX_SIMD_WIDTH;

    threadgroup float dtX[CS * HD];
    threadgroup float sam[NSG * TC * CS];
    threadgroup float tiles[NSG * 2 * TC * TC];
    threadgroup float Lcum[CS];
    threadgroup float exp_Lcum[CS];
    threadgroup float sdecay[CS];
    threadgroup float sg_scratch[NSG];

    threadgroup float* sam_rows = sam   + sgitg * TC * CS;
    threadgroup float* tile0    = tiles + sgitg * 2 * TC * TC;
    threadgroup float* tile1    = tile0 + TC * TC;

    const int slab = int(tgpig.x);
    const int h    = int(tgpig.y);
    const int seq  = int(tgpig.z);

    const int P  = p.head_dim;
    const int N  = p.d_state;
    const int L  = p.n_tok;
    const int g  = h / p.heads_per_group;
    const int p0 = slab * HD;
    const int pw = min(int(HD), P - p0);          // channels this threadgroup owns

    const int src_seq = p.has_ids ? ids[seq] : seq;
    const float A_h   = A[h * p.s_a];

    // Base pointers, already advanced past everything constant for this (seq, head).
    device const float* Xp  = x   + seq * p.seq_x  + h * p.s_x[1];
    device const float* DTp = dtr + seq * p.seq_dt + h * p.s_dt[1];
    device const float* Bp  = Bt  + seq * p.seq_b  + g * p.s_b[1];
    device const float* Cp  = Ct  + seq * p.seq_c  + g * p.s_c[1];
    device       float* Yp  = y   + seq * p.seq_y  + h * p.s_y[1];

    device const float* s_first = s0 + src_seq * p.seq_s + h * p.s_s[0];
    device       float* s_out   = s1 + seq     * p.seq_s + h * p.s_s[0];

    for (int t0 = 0; t0 < L; t0 += CS) {
        const short tlen = short(min(int(CS), L - t0));

        // Chunks after the first read the state this kernel wrote last iteration.
        device const float* s_in = (t0 == 0) ? s_first : s_out;

        // -------------------------------------------------------------------
        // dtX[j][c] = softplus(dt_j) * x[j][c], and the cumulative log-decay.
        // Rows beyond tlen are zeroed so that a ragged chunk contributes
        // nothing through the GEMMs rather than needing a separate code path.
        // -------------------------------------------------------------------
        for (int idx = tiitg; idx < CS * HD; idx += TGS) {
            const int t = idx / HD;
            const int c = idx % HD;
            float v = 0.0f;
            if (t < tlen && c < pw) {
                const float dt_t = helix_dt(p, DTp[(t0 + t) * p.s_dt[0]]);
                v = dt_t * Xp[(t0 + t) * p.s_x[0] + (p0 + c) * p.s_x[2]];
            }
            dtX[idx] = v;
        }

        {
            const float a = (tiitg < tlen) ? A_h * helix_dt(p, DTp[(t0 + tiitg) * p.s_dt[0]])
                                           : 0.0f;
            helix_chunk_prefix_sum(a, tiitg, sgitg, tiisg, ushort(tlen), Lcum, sg_scratch);
        }

        if (tiitg < tlen) {
            // Lcum is a sum of A_h*dt with A_h < 0, so every exponent here is
            // <= 0: these can underflow to zero (the state is genuinely
            // forgotten) but can never overflow.
            exp_Lcum[tiitg] = exp(Lcum[tiitg]);
            sdecay[tiitg]   = exp(Lcum[tlen - 1] - Lcum[tiitg]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float L_end       = Lcum[tlen - 1];
        const float chunk_decay = exp(L_end);

        if (tlen == CS) {
            // ---------------------------------------------------------------
            // Full chunk: everything on the matrix unit.
            // ---------------------------------------------------------------
            for (short ib = sgitg; ib < CS / TC; ib += NSG) {
                // G1: M_raw = C_chunk . B_chunk^T, one 8 x CS row-strip per
                // simdgroup, masked and decayed in place. Built once and reused
                // across every channel tile below -- this amortization is the
                // whole point of the chunked formulation.
                for (short jb = 0; jb <= ib; ++jb) {
                    simdgroup_float8x8 cb = make_filled_simdgroup_matrix<float, 8>(0.0f);
                    for (int k0 = 0; k0 < N; k0 += TC) {
                        simdgroup_float8x8 mc, mb;
                        simdgroup_load(mc, Cp + (t0 + ib * TC) * p.s_c[0] + k0, p.s_c[0]);
                        simdgroup_load(mb, Bp + (t0 + jb * TC) * p.s_b[0] + k0, p.s_b[0], 0, true);
                        simdgroup_multiply_accumulate(cb, mc, mb, cb);
                    }

                    threadgroup float* dst = sam_rows + jb * TC;
                    simdgroup_store(cb, dst, CS);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    for (short e = tiisg; e < TC * TC; e += HELIX_SIMD_WIDTH) {
                        const short ri = e / TC, rj = e % TC;
                        const short i  = ib * TC + ri, j = jb * TC + rj;
                        dst[ri * CS + rj] = (j <= i)
                            ? dst[ri * CS + rj] * exp(Lcum[i] - Lcum[j])
                            : 0.0f;
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }

                // G2 + G3 per 8-wide channel tile.
                for (short ch = 0; ch < HD / TC; ++ch) {
                    if (ch * TC >= pw) break;

                    simdgroup_float8x8 y_diag  = make_filled_simdgroup_matrix<float, 8>(0.0f);
                    simdgroup_float8x8 y_inter = make_filled_simdgroup_matrix<float, 8>(0.0f);

                    for (short jb = 0; jb <= ib; ++jb) {
                        simdgroup_float8x8 ms, md;
                        simdgroup_load(ms, sam_rows + jb * TC, CS);
                        simdgroup_load(md, dtX + jb * TC * HD + ch * TC, HD);
                        simdgroup_multiply_accumulate(y_diag, ms, md, y_diag);
                    }

                    for (int k0 = 0; k0 < N; k0 += TC) {
                        simdgroup_float8x8 mc, mst;
                        simdgroup_load(mc,  Cp + (t0 + ib * TC) * p.s_c[0] + k0, p.s_c[0]);
                        simdgroup_load(mst, s_in + (p0 + ch * TC) * p.s_s[1] + k0,
                                       p.s_s[1], 0, true);
                        simdgroup_multiply_accumulate(y_inter, mc, mst, y_inter);
                    }

                    simdgroup_store(y_diag,  tile0, TC);
                    simdgroup_store(y_inter, tile1, TC);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    for (short e = tiisg; e < TC * TC; e += HELIX_SIMD_WIDTH) {
                        const short ri = e / TC, ci = e % TC;
                        const int   c  = ch * TC + ci;
                        if (c >= pw) continue;
                        const int t = t0 + ib * TC + ri;
                        Yp[t * p.s_y[0] + (p0 + c) * p.s_y[2]] =
                            tile0[e] + exp_Lcum[ib * TC + ri] * tile1[e];
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        } else {
            // ---------------------------------------------------------------
            // Ragged tail chunk. At most one per sequence and at most CS-1
            // tokens, so this stays scalar rather than growing a masked MMA
            // path. Parallelized over t; the C.B dot product is hoisted out of
            // the channel loop so the cost is O(tlen*(N + pw)) per thread, not
            // O(tlen*N*pw).
            // ---------------------------------------------------------------
            for (int t = tiitg; t < tlen; t += TGS) {
                device const float* Crow = Cp + (t0 + t) * p.s_c[0];
                const float exp_Lt = exp_Lcum[t];

                for (int c = 0; c < pw; ++c) {
                    float inter = 0.0f;
                    for (int n = 0; n < N; ++n)
                        inter += Crow[n * p.s_c[2]] * s_in[(p0 + c) * p.s_s[1] + n * p.s_s[2]];
                    Yp[(t0 + t) * p.s_y[0] + (p0 + c) * p.s_y[2]] = exp_Lt * inter;
                }

                for (int j = 0; j <= t; ++j) {
                    device const float* Brow = Bp + (t0 + j) * p.s_b[0];
                    float cb = 0.0f;
                    for (int n = 0; n < N; ++n)
                        cb += Crow[n * p.s_c[2]] * Brow[n * p.s_b[2]];
                    const float w = cb * exp(Lcum[t] - Lcum[j]);
                    for (int c = 0; c < pw; ++c)
                        Yp[(t0 + t) * p.s_y[0] + (p0 + c) * p.s_y[2]] += w * dtX[j * HD + c];
                }
            }
        }

        // All reads of s_in must complete before any thread overwrites s_out;
        // they alias for every chunk after the first. Week 3 removes this
        // barrier entirely by keeping the state in threadgroup memory.
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // -------------------------------------------------------------------
        // G4 + state update. Left scalar here to match upstream exactly; moved
        // onto the matrix unit in Week 3 with the drift test as the guard.
        // -------------------------------------------------------------------
        for (int idx = tiitg; idx < pw * N; idx += TGS) {
            const int c  = idx / N;
            const int n  = idx % N;
            float acc = 0.0f;
            for (short j = 0; j < tlen; ++j)
                acc += sdecay[j] * Bp[(t0 + j) * p.s_b[0] + n * p.s_b[2]] * dtX[j * HD + c];
            s_out[(p0 + c) * p.s_s[1] + n * p.s_s[2]] =
                chunk_decay * s_in[(p0 + c) * p.s_s[1] + n * p.s_s[2]] + acc;
        }

        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
