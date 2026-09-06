// scan_chunked_ref.cpp -- the reassociated (SSD) formulation, in fp64.
//
// This is a deliberate algebraic mirror of what the Metal kernels do: the same
// four GEMMs, the same intermediates (dtX, M, Lcum), the same order. Its only
// purpose is to fail loudly if the reassociation itself is wrong, before any
// MSL exists to blame.
//
//   G1  M_raw = C_chunk . B_chunk^T          (C x N)(N x C) -> C x C
//   G2  Y_intra = tril(M) . dtX              (C x C)(C x P) -> C x P
//   G3  Y_inter = C_chunk . S_prev           (C x N)(N x P) -> C x P
//   G4  dS = B_scaled^T . dtX                (N x C)(C x P) -> N x P
//
// where  M[t,j] = M_raw[t,j] * exp(Lcum_t - Lcum_j)  for j <= t, else 0.

#include "helix/reference.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace helix::ref {

namespace {

inline double softplus(double v) { return v > 20.0 ? v : std::log1p(std::exp(v)); }

}  // namespace

void scan_chunked(const helix_scan_desc& d,
                  const float* s0, const float* x, const float* dt, const float* A,
                  const float* B, const float* C, const int32_t* ids,
                  float* y, float* s1) {
    // The reassociation factors exp(Lcum_t - Lcum_j) out of the state
    // dimension, which is only legal when the decay is one scalar per
    // (head, timestep). Diagonal-A admits no such factorization.
    assert(d.a_form == HELIX_A_SCALAR_PER_HEAD);

    const int64_t H = d.n_head, P = d.head_dim, N = d.d_state, L = d.n_tok;
    const int64_t CK = d.chunk > 0 ? d.chunk : 64;
    const int64_t heads_per_group = d.n_head / d.n_group;

    std::vector<double> S(static_cast<size_t>(P * N));       // S[p][n]
    std::vector<double> dtX(static_cast<size_t>(CK * P));    // dtX[j][p]
    std::vector<double> M(static_cast<size_t>(CK * CK));     // M[t][j]
    std::vector<double> Lcum(static_cast<size_t>(CK));
    std::vector<double> dtv(static_cast<size_t>(CK));

    for (int64_t seq = 0; seq < d.n_seq; ++seq) {
        const int64_t src_seq = ids ? ids[seq] : seq;

        for (int64_t h = 0; h < H; ++h) {
            const int64_t g    = h / heads_per_group;
            const double  A_h  = static_cast<double>(A[h * d.s_a]);

            {
                const float* s_in = s0 + src_seq * d.seq_s + h * d.s_s[0];
                for (int64_t p = 0; p < P; ++p)
                    for (int64_t n = 0; n < N; ++n)
                        S[p * N + n] = static_cast<double>(s_in[p * d.s_s[1] + n * d.s_s[2]]);
            }

            for (int64_t t0 = 0; t0 < L; t0 += CK) {
                const int64_t tlen = std::min<int64_t>(CK, L - t0);

                // --- dt, cumulative log-decay, dtX -------------------------
                double run = 0.0;
                for (int64_t j = 0; j < tlen; ++j) {
                    double v = softplus(static_cast<double>(
                        dt[seq * d.seq_dt + (t0 + j) * d.s_dt[0] + h * d.s_dt[1]]));
                    v = std::clamp(v, static_cast<double>(d.dt_min), static_cast<double>(d.dt_max));
                    dtv[j] = v;
                    run += A_h * v;
                    Lcum[j] = run;
                    for (int64_t p = 0; p < P; ++p)
                        dtX[j * P + p] = v * static_cast<double>(
                            x[seq * d.seq_x + (t0 + j) * d.s_x[0] + h * d.s_x[1] + p * d.s_x[2]]);
                }
                const double L_end = Lcum[tlen - 1];

                // --- G1: M_raw = C . B^T, then mask and apply decay --------
                for (int64_t t = 0; t < tlen; ++t) {
                    const float* Ct = C + seq * d.seq_c + (t0 + t) * d.s_c[0] + g * d.s_c[1];
                    for (int64_t j = 0; j < tlen; ++j) {
                        if (j > t) { M[t * CK + j] = 0.0; continue; }
                        const float* Bj = B + seq * d.seq_b + (t0 + j) * d.s_b[0] + g * d.s_b[1];
                        double cb = 0.0;
                        for (int64_t n = 0; n < N; ++n)
                            cb += static_cast<double>(Ct[n * d.s_c[2]]) *
                                  static_cast<double>(Bj[n * d.s_b[2]]);
                        // Lcum_t - Lcum_j <= 0 for j <= t (A_h < 0), so this
                        // factor is in (0, 1]: it can underflow, never overflow.
                        M[t * CK + j] = cb * std::exp(Lcum[t] - Lcum[j]);
                    }
                }

                // --- G2 + G3: y = tril(M).dtX + exp(Lcum_t)*(C.S_prev) -----
                for (int64_t t = 0; t < tlen; ++t) {
                    const float*  Ct     = C + seq * d.seq_c + (t0 + t) * d.s_c[0] + g * d.s_c[1];
                    const double  exp_Lt = std::exp(Lcum[t]);
                    for (int64_t p = 0; p < P; ++p) {
                        double intra = 0.0;
                        for (int64_t j = 0; j <= t; ++j)
                            intra += M[t * CK + j] * dtX[j * P + p];

                        double inter = 0.0;
                        for (int64_t n = 0; n < N; ++n)
                            inter += static_cast<double>(Ct[n * d.s_c[2]]) * S[p * N + n];

                        y[seq * d.seq_y + (t0 + t) * d.s_y[0] + h * d.s_y[1] + p * d.s_y[2]] =
                            static_cast<float>(intra + exp_Lt * inter);
                    }
                }

                // --- G4 + state update ------------------------------------
                const double chunk_decay = std::exp(L_end);
                for (int64_t p = 0; p < P; ++p) {
                    for (int64_t n = 0; n < N; ++n) {
                        double ds = 0.0;
                        for (int64_t j = 0; j < tlen; ++j) {
                            const float* Bj =
                                B + seq * d.seq_b + (t0 + j) * d.s_b[0] + g * d.s_b[1];
                            ds += std::exp(L_end - Lcum[j]) *
                                  static_cast<double>(Bj[n * d.s_b[2]]) * dtX[j * P + p];
                        }
                        S[p * N + n] = chunk_decay * S[p * N + n] + ds;
                    }
                }
            }

            float* s_out = s1 + seq * d.seq_s + h * d.s_s[0];
            for (int64_t p = 0; p < P; ++p)
                for (int64_t n = 0; n < N; ++n)
                    s_out[p * d.s_s[1] + n * d.s_s[2]] = static_cast<float>(S[p * N + n]);
        }
    }
}

}  // namespace helix::ref
