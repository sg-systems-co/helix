#pragma once
// harness.hpp -- zero-dependency test scaffolding.
//
// Inputs are generated from a fixed-seed xorshift rather than <random> so that
// a failure reproduces byte-for-byte on any machine.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "helix/helix.h"

namespace helix::test {

// ---------------------------------------------------------------------------
// Deterministic input generation
// ---------------------------------------------------------------------------

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed = 0x9E3779B97F4A7C15ull) : s(seed ? seed : 1) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    // Uniform in [lo, hi).
    float uniform(float lo, float hi) {
        const float u = static_cast<float>(next() >> 40) / 16777216.0f;  // [0,1)
        return lo + u * (hi - lo);
    }
};

inline void fill_uniform(std::vector<float>& v, Rng& rng, float lo, float hi) {
    for (auto& e : v) e = rng.uniform(lo, hi);
}

// A must be strictly negative for the recurrence to decay; real models
// parameterize it as A = -exp(A_log), which is what we mimic here.
inline void fill_A(std::vector<float>& v, Rng& rng) {
    for (auto& e : v) e = -std::exp(rng.uniform(-2.0f, 0.5f));
}

// ---------------------------------------------------------------------------
// Error metrics
// ---------------------------------------------------------------------------

struct ErrorStats {
    double  max_abs     = 0.0;
    double  max_rel     = 0.0;
    double  rms         = 0.0;
    // max over elements of |a-b| / (abs_tol + rel_tol*|b|). Pass iff <= 1.
    // This is the numpy allclose criterion applied per element; taking the max
    // of |a-b| and the max of the relative error *separately* and OR-ing them
    // is far weaker and silently accepts real regressions.
    double  worst_ratio = 0.0;
    int64_t worst_at    = -1;
    bool    finite      = true;
};

inline ErrorStats compare(const std::vector<float>& got, const std::vector<float>& ref,
                          double rel_tol, double abs_tol) {
    ErrorStats st;
    double sq = 0.0;
    const size_t n = got.size() < ref.size() ? got.size() : ref.size();
    for (size_t i = 0; i < n; ++i) {
        const double a = got[i], b = ref[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            st.finite = false; st.worst_at = (int64_t)i; break;
        }
        const double abs_err = std::fabs(a - b);
        const double rel_err = abs_err / (std::fabs(b) + 1e-12);
        const double ratio   = abs_err / (abs_tol + rel_tol * std::fabs(b));
        if (abs_err > st.max_abs) st.max_abs = abs_err;
        if (rel_err > st.max_rel) st.max_rel = rel_err;
        if (ratio   > st.worst_ratio) { st.worst_ratio = ratio; st.worst_at = (int64_t)i; }
        sq += abs_err * abs_err;
    }
    st.rms = n ? std::sqrt(sq / static_cast<double>(n)) : 0.0;
    return st;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

class Suite {
public:
    explicit Suite(std::string name) : name_(std::move(name)) {}

    void check(bool ok, const std::string& what) {
        ++total_;
        if (ok) { std::printf("  ok    %s\n", what.c_str()); }
        else    { ++failed_; std::printf("  FAIL  %s\n", what.c_str()); }
    }

    // Tolerances live in the ErrorStats (compare() applied them per element);
    // this only reports.
    void check_error(const ErrorStats& st, const std::string& what) {
        const bool ok = st.finite && st.worst_ratio <= 1.0;
        ++total_;
        if (ok) {
            std::printf("  ok    %-46s abs=%.3e rms=%.3e ratio=%.3f\n",
                        what.c_str(), st.max_abs, st.rms, st.worst_ratio);
        } else {
            ++failed_;
            std::printf("  FAIL  %-46s rel=%.3e abs=%.3e rms=%.3e ratio=%.2f%s @%lld\n",
                        what.c_str(), st.max_rel, st.max_abs, st.rms, st.worst_ratio,
                        st.finite ? "" : " NON-FINITE", (long long)st.worst_at);
        }
    }

    int finish() const {
        std::printf("[%s] %d/%d passed\n", name_.c_str(), total_ - failed_, total_);
        return failed_ == 0 ? 0 : 1;
    }

private:
    std::string name_;
    int total_  = 0;
    int failed_ = 0;
};

// ---------------------------------------------------------------------------
// A complete, self-consistent problem instance
// ---------------------------------------------------------------------------

// How much of the state survives a chunk.
//
// This is not a cosmetic knob. With SHORT_MEMORY the per-chunk decay measures
// 1e-10 to 1e-18: the state is annihilated inside a single chunk, the
// inter-chunk carry contributes nothing to the output, and a test built on it
// silently cannot detect a bug in that carry -- verified by injecting a 1.002x
// error into the scan and watching every number stay identical.
//
// LONG_MEMORY matches how real Mamba-2 checkpoints are parameterised
// (softplus(dt_bias) initialised into roughly [0.001, 0.1]), which leaves a
// per-chunk decay near 0.5 and makes adjacent chunks genuinely coupled.
enum class Memory {
    SHORT,  // aggressive decay; exercises the intra-chunk math
    LONG,   // realistic; exercises the inter-chunk carry
};

struct Problem {
    helix_scan_desc d{};
    std::vector<float>   s0, x, dt, A, B, C;
    std::vector<int32_t> ids;

    static Problem make(int32_t n_tok, int32_t n_seq, int32_t n_head,
                        int32_t head_dim, int32_t d_state, int32_t n_group,
                        int32_t chunk, uint64_t seed,
                        Memory mem = Memory::SHORT) {
        Problem p;
        helix_scan_desc_init(&p.d, n_tok, n_seq, n_head, head_dim, d_state, n_group);
        p.d.chunk = chunk;

        Rng rng(seed);
        p.s0.resize((size_t)n_seq * n_head * head_dim * d_state);
        p.x .resize((size_t)n_seq * n_tok * n_head * head_dim);
        p.dt.resize((size_t)n_seq * n_tok * n_head);
        p.A .resize((size_t)n_head);
        p.B .resize((size_t)n_seq * n_tok * n_group * d_state);
        p.C .resize((size_t)n_seq * n_tok * n_group * d_state);
        p.ids.resize((size_t)n_seq);
        for (int32_t i = 0; i < n_seq; ++i) p.ids[i] = i;

        fill_uniform(p.s0, rng, -0.5f, 0.5f);
        fill_uniform(p.x,  rng, -1.0f, 1.0f);
        if (mem == Memory::LONG) {
            // softplus lands dt in ~[0.0009, 0.011]; with |A| ~ 1.4 that is a
            // per-chunk decay near 0.5.
            fill_uniform(p.dt, rng, -7.0f, -4.5f);
            for (auto& e : p.A) e = -std::exp(rng.uniform(-0.5f, 1.0f));
        } else {
            fill_uniform(p.dt, rng, -4.0f, 2.0f);
            fill_A(p.A, rng);
        }
        // B and C are scaled by 1/sqrt(d_state) so that the C.B^T dot products
        // stay O(1) and fp16 range questions are realistic rather than rigged.
        const float sc = 1.0f / std::sqrt((float)d_state);
        fill_uniform(p.B, rng, -sc, sc);
        fill_uniform(p.C, rng, -sc, sc);
        return p;
    }

    size_t y_elems() const {
        return (size_t)d.n_seq * d.n_tok * d.n_head * d.head_dim;
    }
    size_t s_elems() const {
        return (size_t)d.n_seq * d.n_head * d.head_dim * d.d_state;
    }
};

}  // namespace helix::test
