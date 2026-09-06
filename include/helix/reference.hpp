#pragma once
// reference.hpp -- CPU ground truth for the HELIX scan.
//
// Two independent implementations, both computing in double:
//
//   helix_ref_scan          the sequential recurrence. This IS the definition
//                           of correct; nothing validates it but inspection.
//   helix_ref_scan_chunked  the chunked/reassociated SSD formulation. Its only
//                           job is to prove the algebra HELIX's kernels
//                           implement, in a debuggable setting, before any MSL
//                           exists. If this disagrees with the sequential
//                           reference the algebra is wrong, not the GPU.
//
// Inputs and outputs are f32 arrays laid out per the helix_scan_desc strides;
// all intermediate arithmetic is f64.

#include <cstdint>
#include <vector>

#include "helix/helix.h"

namespace helix::ref {

// Sequential recurrence, per (seq, head, chan):
//
//   dt_t   = softplus(dt_raw[t,h]) clamped to [dt_min, dt_max]
//   dA_t   = exp(A[h] * dt_t)                       (scalar-A)
//            exp(A[h,n] * dt_t)                     (diagonal-A)
//   s[n]  <- dA_t * s[n] + dt_t * B[t,g,n] * x[t,h,p]
//   y[t]   = sum_n C[t,g,n] * s[n]
void scan(const helix_scan_desc& d,
          const float* s0, const float* x, const float* dt, const float* A,
          const float* B, const float* C, const int32_t* ids,
          float* y, float* s1);

// Chunked / reassociated SSD. Only valid for HELIX_A_SCALAR_PER_HEAD -- the
// reassociation into a dense M matrix relies on the decay being a scalar per
// (head, timestep) so that exp(Lcum_t - Lcum_j) factors out of the state
// dimension. Diagonal-A has no such factorization and stays sequential.
void scan_chunked(const helix_scan_desc& d,
                  const float* s0, const float* x, const float* dt, const float* A,
                  const float* B, const float* C, const int32_t* ids,
                  float* y, float* s1);

// Per-chunk-boundary state snapshots, used by the numerical-drift test to
// assert that error stays bounded across many chunks rather than compounding.
// `out_states` receives n_chunks entries of (n_head * head_dim * d_state).
void scan_with_snapshots(const helix_scan_desc& d,
                         const float* s0, const float* x, const float* dt, const float* A,
                         const float* B, const float* C, const int32_t* ids,
                         float* y, float* s1,
                         int32_t chunk, std::vector<std::vector<double>>* out_states);

}  // namespace helix::ref
