// helix_scan_multipass.metal -- the 3-pass chunk-parallel decomposition.
//
// The single-pass kernel dispatches only (n_pslab x n_head x n_seq) threadgroups
// and walks chunks serially inside each. Measured on an M5 Max, that leaves the
// GPU badly underfed: holding the work per threadgroup fixed and merely raising
// the threadgroup count from 64 to 512 took the same kernel from 513 to 890
// GFLOP/s. This file buys that 1.73x for a single sequence, where there is no
// batch dimension to harvest it from.
//
// The recurrence over chunks is an associative scan:
//
//     (a1, b1) . (a2, b2) = (a1*a2, a2*b1 + b2)
//
// with a = exp(Lcum_end) a scalar per (chunk, head) and b = dS the chunk's own
// state contribution. So:
//
//   Pass A  helix_chunk_state_f32   every chunk computes its dS independently
//   Pass B  helix_state_scan_f32    scan the (decay, dS) pairs -> S_init[c]
//   Pass C  helix_chunk_out_f32     every chunk computes its output from S_init
//
// A and C are embarrassingly parallel across chunks. B is ~1% of the work
// (n_cg * head_dim * d_state per head), so it is a serial walk in registers
// rather than a Blelloch tree -- building the tree before B shows up in a
// profile would be pure ceremony.
//
// Scratch is bounded by super-chunk tiling: the host loops serially over groups
// of at most n_cg chunks, carrying state between groups through s1.
//
// Everything here is fp32 and the state still lives in device memory, exactly
// like the single-pass kernel. That is deliberate -- it isolates the occupancy
// change from the precision and memory-residency work that follows.

#include "helix_common.h"

using namespace metal;

// Grid for all three passes:
//   x = chunk within the group (1 for the scan pass)
//   y = head
//   z = seq * n_pslab + slab
#define HELIX_MP_DECODE_Z                              \
    const int slab = int(tgpig.z) % p.n_pslab;         \
    const int seq  = int(tgpig.z) / p.n_pslab;         \
    const int h    = int(tgpig.y);                     \
    const int p0   = slab * HELIX_HD;                  \
    const int pw   = min(int(HELIX_HD), p.head_dim - p0)

// ---------------------------------------------------------------------------
// Pass A -- per-chunk state contribution
// ---------------------------------------------------------------------------
kernel void helix_chunk_state_f32(
        constant HelixScanParams& p      [[buffer(0)]],
        device const float*       s0     [[buffer(1)]],
        device const float*       x      [[buffer(2)]],
        device const float*       dtr    [[buffer(3)]],
        device const float*       A      [[buffer(4)]],
        device const float*       Bt     [[buffer(5)]],
        device const float*       Ct     [[buffer(6)]],
        device const int*         ids    [[buffer(7)]],
        device       float*       dS     [[buffer(10)]],
        device       float*       logdec [[buffer(11)]],
        device const float*       Btail  [[buffer(15)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {

    constexpr ushort TGS = HELIX_NSG * HELIX_SIMD_WIDTH;

    threadgroup float dtX[HELIX_CS * HELIX_HD];
    threadgroup float Lcum[HELIX_CS];
    threadgroup float exp_Lcum[HELIX_CS];
    threadgroup float sdecay[HELIX_CS];
    threadgroup float sg_scratch[HELIX_NSG];

    HELIX_MP_DECODE_Z;

    const int cg = int(tgpig.x);
    if (cg >= p.n_cg) return;

    const int t0 = (p.chunk_base + cg) * HELIX_CS;
    if (t0 >= p.n_tok) return;
    const short tlen = short(min(int(HELIX_CS), p.n_tok - t0));

    const int   N   = p.d_state;
    const int   g   = h / p.heads_per_group;
    const float A_h = A[h * p.s_a];

    device const float* Xp  = x   + seq * p.seq_x  + h * p.s_x[1];
    device const float* DTp = dtr + seq * p.seq_dt + h * p.s_dt[1];
    device const float* Bp  = Bt  + seq * p.seq_b  + g * p.s_b[1];

    const float L_end = helix_chunk_preamble(p, DTp, Xp, t0, tlen, p0, pw, A_h,
                                             tiitg, sgitg, tiisg, TGS,
                                             dtX, Lcum, exp_Lcum, sdecay, sg_scratch);

    // One writer per (seq, chunk, head): the decay is a scalar, so only the
    // first slab records it.
    if (slab == 0 && tiitg == 0)
        logdec[seq * p.ld_seq + cg * p.ld_cg + h] = L_end;

    // dS[n][c] = sum_j exp(L_end - Lcum_j) * B[j][n] * dtX[j][c]
    //
    // Folding the per-timestep decay into dtX turns this into a plain GEMM,
    //
    //     dS = B^T . dtXs      (N x C)(C x P) -> N x P
    //
    // because exp(L_end - Lcum_j) scales an entire j-row. Pass A never computes
    // y, so unlike Pass C it is free to scale dtX in place.
    //
    // This is the kernel's single biggest cost: profiled at L=8192 it was 65%
    // of total runtime while running entirely on the scalar ALU.
    for (int idx = tiitg; idx < HELIX_CS * HELIX_HD; idx += TGS) {
        const int t = idx / HELIX_HD;
        dtX[idx] *= (t < tlen) ? sdecay[t] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* dsp = dS + seq * p.sc_seq + cg * p.sc_cg + h * N * p.head_dim;

    // A ragged chunk reads its B rows from the zero-padded tail copy instead of
    // the caller's buffer, so the 8-row tile loads below stay in bounds and
    // there is exactly one code path for every chunk.
    device const float* Bchunk;
    int b_stride;
    if (tlen == HELIX_CS) {
        Bchunk   = Bp + t0 * p.s_b[0];
        b_stride = p.s_b[0];
    } else {
        Bchunk   = Btail + seq * p.tail_seq + g * N;
        b_stride = p.tail_stride;
    }

    for (int nb = sgitg; nb < N / HELIX_TC; nb += HELIX_NSG) {
        for (int cb = 0; cb * HELIX_TC < pw; ++cb) {
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
            for (int j0 = 0; j0 < HELIX_CS; j0 += HELIX_TC) {
                simdgroup_float8x8 mb, md;
                // Transposed load gives mb[r][k] = B[j0+k][nb*8+r].
                simdgroup_load(mb, Bchunk + j0 * b_stride + nb * HELIX_TC,
                               b_stride, 0, true);
                simdgroup_load(md, dtX + j0 * HELIX_HD + cb * HELIX_TC, HELIX_HD);
                simdgroup_multiply_accumulate(acc, mb, md, acc);
            }
            simdgroup_store(acc, dsp + nb * HELIX_TC * p.head_dim + p0 + cb * HELIX_TC,
                            p.head_dim);
        }
    }
}

// ---------------------------------------------------------------------------
// Pass B -- inter-chunk associative scan
// ---------------------------------------------------------------------------
kernel void helix_state_scan_f32(
        constant HelixScanParams& p      [[buffer(0)]],
        device const float*       s0     [[buffer(1)]],
        device const int*         ids    [[buffer(7)]],
        device       float*       s1     [[buffer(9)]],
        device       float*       dS     [[buffer(10)]],
        device const float*       logdec [[buffer(11)]],
        device       bfloat*      Sbf    [[buffer(14)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {

    constexpr ushort TGS = HELIX_NSG * HELIX_SIMD_WIDTH;

    threadgroup float ld[HELIX_MAX_CG];

    HELIX_MP_DECODE_Z;

    const int N  = p.d_state;
    const int P  = p.head_dim;

    for (int i = tiitg; i < p.n_cg; i += TGS)
        ld[i] = logdec[seq * p.ld_seq + i * p.ld_cg + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int src_seq = p.has_ids ? ids[seq] : seq;

    // The first group starts from the caller's s0 (through the ids indirection);
    // later groups continue from whatever the previous group left in s1.
    device const float* carry_in = p.first_group
        ? (s0 + src_seq * p.seq_s + h * p.s_s[0])
        : (s1 + seq     * p.seq_s + h * p.s_s[0]);
    device float* carry_out = s1 + seq * p.seq_s + h * p.s_s[0];

    device float* base = dS + seq * p.sc_seq + h * N * P;

    for (int idx = tiitg; idx < pw * N; idx += TGS) {
        // Decomposed state-major to match the dS layout, so adjacent threads
        // touch adjacent channels and the dS traffic coalesces. Indexing this
        // the other way round costs ~8x here: dS is read and written once per
        // chunk in the walk below, so it dominates the two carry accesses.
        const int n = idx / pw;
        const int c = idx % pw;

        // Each thread owns one (chan, state) slot for the whole walk, so the
        // read-then-overwrite below has no cross-thread hazard and needs no
        // barrier. carry_in and carry_out alias for every group after the
        // first; that is safe for the same reason.
        float running = carry_in[(p0 + c) * p.s_s[1] + n * p.s_s[2]];

        device bfloat* bf_base = Sbf ? (Sbf + seq * p.sc_seq + h * N * P) : nullptr;

        for (int gi = 0; gi < p.n_cg; ++gi) {
            device float* slot = base + gi * p.sc_cg + n * P + (p0 + c);
            const float ds = *slot;
            *slot = running;                       // now S_init for chunk gi
            // The MPP path consumes the initial state as a bf16 operand. The
            // carry itself stays fp32 -- only the copy Pass C reads is narrowed.
            if (bf_base) bf_base[gi * p.sc_cg + n * P + (p0 + c)] = bfloat(running);
            // One chunk's decay per step, never a product of many, so this
            // needs no log-space accumulation. Underflow to zero is the correct
            // answer: the state has genuinely been forgotten.
            running = exp(ld[gi]) * running + ds;
        }

        carry_out[(p0 + c) * p.s_s[1] + n * p.s_s[2]] = running;
    }
}

// ---------------------------------------------------------------------------
// Pass C -- per-chunk output
// ---------------------------------------------------------------------------
kernel void helix_chunk_out_f32(
        constant HelixScanParams& p   [[buffer(0)]],
        device const float*       x   [[buffer(2)]],
        device const float*       dtr [[buffer(3)]],
        device const float*       A   [[buffer(4)]],
        device const float*       Bt  [[buffer(5)]],
        device const float*       Ct  [[buffer(6)]],
        device       float*       y     [[buffer(8)]],
        device const float*       dS    [[buffer(10)]],
        device const float*       Btail [[buffer(15)]],
        device const float*       Ctail [[buffer(16)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {

    constexpr short  CS  = HELIX_CS;
    constexpr short  HD  = HELIX_HD;
    constexpr short  TC  = HELIX_TC;
    constexpr short  NSG = HELIX_NSG;
    constexpr ushort TGS = NSG * HELIX_SIMD_WIDTH;

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

    HELIX_MP_DECODE_Z;

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
    device const float* Bp  = Bt  + seq * p.seq_b  + g * p.s_b[1];
    device const float* Cp  = Ct  + seq * p.seq_c  + g * p.s_c[1];
    device       float* Yp  = y   + seq * p.seq_y  + h * p.s_y[1];

    helix_chunk_preamble(p, DTp, Xp, t0, tlen, p0, pw, A_h,
                         tiitg, sgitg, tiisg, TGS,
                         dtX, Lcum, exp_Lcum, sdecay, sg_scratch);

    // Pass B left the state entering this chunk here, in (d_state, head_dim)
    // order -- so G3 below needs no transposed operand.
    device const float* s_in = dS + seq * p.sc_seq + cg * p.sc_cg + h * N * p.head_dim;

    // A ragged chunk reads B and C from the zero-padded tail copy, so the 8-row
    // tile loads stay in bounds and every chunk takes the same path. The old
    // scalar tail branch measured 843 us at L=96 against upstream's 246 -- the
    // padding costs one small copy kernel and removes the notch entirely.
    device const float* Cchunk;
    device const float* Bchunk;
    int c_stride, b_stride;
    if (tlen == CS) {
        Cchunk = Cp + t0 * p.s_c[0];  c_stride = p.s_c[0];
        Bchunk = Bp + t0 * p.s_b[0];  b_stride = p.s_b[0];
    } else {
        Cchunk = Ctail + seq * p.tail_seq + g * N;  c_stride = p.tail_stride;
        Bchunk = Btail + seq * p.tail_seq + g * N;  b_stride = p.tail_stride;
    }

    for (short ib = sgitg; ib < CS / TC; ib += NSG) {
        // G1: M_raw = C_chunk . B_chunk^T, masked and decayed in place.
        for (short jb = 0; jb <= ib; ++jb) {
            simdgroup_float8x8 cb = make_filled_simdgroup_matrix<float, 8>(0.0f);
            for (int k0 = 0; k0 < N; k0 += TC) {
                simdgroup_float8x8 mc, mb;
                simdgroup_load(mc, Cchunk + (ib * TC) * c_stride + k0, c_stride);
                simdgroup_load(mb, Bchunk + (jb * TC) * b_stride + k0, b_stride, 0, true);
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
                simdgroup_load(mc,  Cchunk + (ib * TC) * c_stride + k0, c_stride);
                simdgroup_load(mst, s_in + k0 * p.head_dim + p0 + ch * TC, p.head_dim);
                simdgroup_multiply_accumulate(y_inter, mc, mst, y_inter);
            }

            simdgroup_store(y_diag,  tile0, TC);
            simdgroup_store(y_inter, tile1, TC);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (short e = tiisg; e < TC * TC; e += HELIX_SIMD_WIDTH) {
                const short ri = e / TC, ci = e % TC;
                const int   c  = ch * TC + ci;
                const short t  = ib * TC + ri;
                // Padded rows produce zeros through the GEMMs but must not be
                // written: they are past the end of y.
                if (c >= pw || t >= tlen) continue;
                Yp[(t0 + t) * p.s_y[0] + (p0 + c) * p.s_y[2]] =
                    tile0[e] + exp_Lcum[t] * tile1[e];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
