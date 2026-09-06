// helix_cast.metal -- dense bf16 staging copies of B and C.
//
// MPP's matmul2d only reaches the M5 Neural Accelerators when both operands are
// bf16: measured on an M5 Max, bf16 x bf16 runs at 65.7 TFLOP/s while
// bf16 x f32 collapses to 15.6, the same as plain f32. So there is no way to
// feed it the caller's f32 tensors directly.
//
// The copy is also densified. The source may be strided (ggml hands over views
// with arbitrary nb[]), while an MPP tensor over the chunk wants a predictable
// (state-contiguous) layout.
//
// Cost is one pass over B and C -- about 100 MB of traffic at L=8192, roughly
// 0.25 ms, against a Pass C that takes several ms.

#include "helix_common.h"

using namespace metal;

kernel void helix_cast_bc_bf16(
        constant HelixScanParams& p   [[buffer(0)]],
        device const float*       Bt  [[buffer(5)]],
        device const float*       Ct  [[buffer(6)]],
        device       bfloat*      Bbf [[buffer(12)]],
        device       bfloat*      Cbf [[buffer(13)]],
        uint3 gid [[thread_position_in_grid]]) {

    const int n = int(gid.x);              // state
    const int t = int(gid.y);              // token
    const int s = int(gid.z);              // sequence
    if (n >= p.d_state || t >= p.n_tok_pad) return;

    // Rows past the end of the sequence are zeroed rather than skipped: Pass C
    // builds a full CS-row tensor over the ragged last chunk, and those rows are
    // masked out of the result but must still be safe to read.
    const bool live = (t < p.n_tok);

    for (int g = 0; g < p.n_group; ++g) {
        const int dst = s * p.bf_seq + t * p.bf_tok + g * p.d_state + n;
        Bbf[dst] = live
            ? bfloat(Bt[s * p.seq_b + t * p.s_b[0] + g * p.s_b[1] + n * p.s_b[2]])
            : bfloat(0.0f);
        Cbf[dst] = live
            ? bfloat(Ct[s * p.seq_c + t * p.s_c[0] + g * p.s_c[1] + n * p.s_c[2]])
            : bfloat(0.0f);
    }
}
