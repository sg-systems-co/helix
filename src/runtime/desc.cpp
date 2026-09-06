// desc.cpp -- descriptor defaults and dense-stride derivation. No Metal here,
// so the reference implementations and their tests build without a GPU.

#include <cmath>
#include <cstring>

#include "helix/helix.h"

extern "C" void helix_scan_desc_init(helix_scan_desc* d,
                                     int32_t n_tok, int32_t n_seq, int32_t n_head,
                                     int32_t head_dim, int32_t d_state, int32_t n_group) {
    std::memset(d, 0, sizeof(*d));

    d->n_tok    = n_tok;
    d->n_seq    = n_seq;
    d->n_head   = n_head;
    d->head_dim = head_dim;
    d->d_state  = d_state;
    d->n_group  = n_group;

    const int64_t L = n_tok, H = n_head, P = head_dim, N = d_state, G = n_group;

    // Dense strides for the layouts documented in helix.h (fastest last).
    d->s_x[0] = H * P;  d->s_x[1] = P;  d->s_x[2] = 1;  d->seq_x  = L * H * P;
    d->s_y[0] = H * P;  d->s_y[1] = P;  d->s_y[2] = 1;  d->seq_y  = L * H * P;
    d->s_b[0] = G * N;  d->s_b[1] = N;  d->s_b[2] = 1;  d->seq_b  = L * G * N;
    d->s_c[0] = G * N;  d->s_c[1] = N;  d->s_c[2] = 1;  d->seq_c  = L * G * N;
    d->s_dt[0] = H;     d->s_dt[1] = 1;                 d->seq_dt = L * H;
    d->s_s[0] = P * N;  d->s_s[1] = N;  d->s_s[2] = 1;  d->seq_s  = H * P * N;

    d->s_a = 1;  // scalar-per-head; callers selecting diagonal-A set this to N

    d->chunk         = 0;  // library chooses
    d->a_form        = HELIX_A_SCALAR_PER_HEAD;
    d->io_dtype      = HELIX_F32;
    // Week 2 ships the fp32 path; the bf16 default arrives with the on-chip
    // state work in Week 3.
    d->compute_dtype = HELIX_F32;
    d->backend       = HELIX_BACKEND_AUTO;
    d->multipass     = false;

    // ggml applies no clamp; lite-ssm does. Disabled by default so HELIX
    // matches the integration target out of the box.
    d->dt_min = 0.0f;
    d->dt_max = INFINITY;
}
