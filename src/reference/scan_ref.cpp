// scan_ref.cpp -- sequential fp64 ground truth. This file defines "correct".

#include "helix/reference.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace helix::ref {

namespace {

inline double softplus(double v) {
    // The x > 20 shortcut matches ggml-metal's kernel exactly; without it the
    // GPU and the reference disagree in the tail even though both are "right".
    return v > 20.0 ? v : std::log1p(std::exp(v));
}

inline double dt_at(const helix_scan_desc& d, const float* dt, int64_t seq, int64_t t, int64_t h) {
    double v = softplus(static_cast<double>(dt[seq * d.seq_dt + t * d.s_dt[0] + h * d.s_dt[1]]));
    if (v < d.dt_min) v = d.dt_min;
    if (v > d.dt_max) v = d.dt_max;
    return v;
}

// A is either one scalar per head, or one value per (head, state).
inline double a_at(const helix_scan_desc& d, const float* A, int64_t h, int64_t n) {
    return static_cast<double>(A[h * d.s_a + (d.a_form == HELIX_A_DIAG_PER_CHANNEL ? n : 0)]);
}

}  // namespace

void scan(const helix_scan_desc& d,
          const float* s0, const float* x, const float* dt, const float* A,
          const float* B, const float* C, const int32_t* ids,
          float* y, float* s1) {
    const int64_t H = d.n_head, P = d.head_dim, N = d.d_state;
    const int64_t heads_per_group = d.n_head / d.n_group;

    std::vector<double> s(static_cast<size_t>(N));

    for (int64_t seq = 0; seq < d.n_seq; ++seq) {
        const int64_t src_seq = ids ? ids[seq] : seq;

        for (int64_t h = 0; h < H; ++h) {
            const int64_t g = h / heads_per_group;

            for (int64_t p = 0; p < P; ++p) {
                const float* s_in = s0 + src_seq * d.seq_s + h * d.s_s[0] + p * d.s_s[1];
                for (int64_t n = 0; n < N; ++n) s[n] = static_cast<double>(s_in[n * d.s_s[2]]);

                for (int64_t t = 0; t < d.n_tok; ++t) {
                    const double dt_t = dt_at(d, dt, seq, t, h);
                    const double x_t  = static_cast<double>(
                        x[seq * d.seq_x + t * d.s_x[0] + h * d.s_x[1] + p * d.s_x[2]]);

                    const float* Bt = B + seq * d.seq_b + t * d.s_b[0] + g * d.s_b[1];
                    const float* Ct = C + seq * d.seq_c + t * d.s_c[0] + g * d.s_c[1];

                    double acc = 0.0;
                    for (int64_t n = 0; n < N; ++n) {
                        const double dA = std::exp(a_at(d, A, h, n) * dt_t);
                        s[n] = dA * s[n] + dt_t * static_cast<double>(Bt[n * d.s_b[2]]) * x_t;
                        acc += static_cast<double>(Ct[n * d.s_c[2]]) * s[n];
                    }

                    y[seq * d.seq_y + t * d.s_y[0] + h * d.s_y[1] + p * d.s_y[2]] =
                        static_cast<float>(acc);
                }

                float* s_out = s1 + seq * d.seq_s + h * d.s_s[0] + p * d.s_s[1];
                for (int64_t n = 0; n < N; ++n) s_out[n * d.s_s[2]] = static_cast<float>(s[n]);
            }
        }
    }
}

// Single pass. The obvious implementation -- re-running scan() truncated to
// each boundary -- is quadratic in n_chunks and takes minutes at L=8192.
void scan_with_snapshots(const helix_scan_desc& d,
                         const float* s0, const float* x, const float* dt, const float* A,
                         const float* B, const float* C, const int32_t* ids,
                         float* y, float* s1,
                         int32_t chunk, std::vector<std::vector<double>>* out_states) {
    if (!out_states || chunk <= 0) {
        scan(d, s0, x, dt, A, B, C, ids, y, s1);
        return;
    }

    const int64_t H = d.n_head, P = d.head_dim, N = d.d_state;
    const int64_t heads_per_group = d.n_head / d.n_group;
    const int32_t n_chunks = (d.n_tok + chunk - 1) / chunk;

    const size_t state_elems =
        static_cast<size_t>(d.n_seq) * d.n_head * d.head_dim * d.d_state;
    out_states->assign(n_chunks, std::vector<double>(state_elems));

    std::vector<double> s(static_cast<size_t>(N));

    for (int64_t seq = 0; seq < d.n_seq; ++seq) {
        const int64_t src_seq = ids ? ids[seq] : seq;

        for (int64_t h = 0; h < H; ++h) {
            const int64_t g = h / heads_per_group;

            for (int64_t p = 0; p < P; ++p) {
                const float* s_in = s0 + src_seq * d.seq_s + h * d.s_s[0] + p * d.s_s[1];
                for (int64_t n = 0; n < N; ++n) s[n] = static_cast<double>(s_in[n * d.s_s[2]]);

                for (int64_t t = 0; t < d.n_tok; ++t) {
                    const double dt_t = dt_at(d, dt, seq, t, h);
                    const double x_t  = static_cast<double>(
                        x[seq * d.seq_x + t * d.s_x[0] + h * d.s_x[1] + p * d.s_x[2]]);

                    const float* Bt = B + seq * d.seq_b + t * d.s_b[0] + g * d.s_b[1];
                    const float* Ct = C + seq * d.seq_c + t * d.s_c[0] + g * d.s_c[1];

                    double acc = 0.0;
                    for (int64_t n = 0; n < N; ++n) {
                        const double dA = std::exp(a_at(d, A, h, n) * dt_t);
                        s[n] = dA * s[n] + dt_t * static_cast<double>(Bt[n * d.s_b[2]]) * x_t;
                        acc += static_cast<double>(Ct[n * d.s_c[2]]) * s[n];
                    }

                    y[seq * d.seq_y + t * d.s_y[0] + h * d.s_y[1] + p * d.s_y[2]] =
                        static_cast<float>(acc);

                    // Snapshot on the last token of each chunk, in the caller's
                    // state layout so it can be compared to s1 directly.
                    if ((t + 1) % chunk == 0 || t + 1 == d.n_tok) {
                        const int32_t c = (int32_t)(t / chunk);
                        double* dst = (*out_states)[c].data() +
                                      seq * d.seq_s + h * d.s_s[0] + p * d.s_s[1];
                        for (int64_t n = 0; n < N; ++n) dst[n * d.s_s[2]] = s[n];
                    }
                }

                float* s_out = s1 + seq * d.seq_s + h * d.s_s[0] + p * d.s_s[1];
                for (int64_t n = 0; n < N; ++n) s_out[n * d.s_s[2]] = static_cast<float>(s[n]);
            }
        }
    }
}

}  // namespace helix::ref
