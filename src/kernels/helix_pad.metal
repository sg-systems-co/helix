// helix_pad.metal -- zero-padded copy of the final ragged chunk's B and C.
//
// The fp32 MMA path loads B and C in 8-row simdgroup tiles. When the last chunk
// holds fewer than HELIX_CS tokens those tiles would read past the end of the
// sequence, which is why the kernel used to branch to a scalar tail. That
// branch computed y_inter with one thread per token and a head_dim-strided
// read, and measured 843 us at L=96 against upstream's 246 -- worse than the
// thing it was meant to accelerate.
//
// Padding the source instead removes the branch entirely: live rows are copied,
// the rest are zeroed, and zeros contribute nothing through the GEMMs. This is
// exactly what the MPP path already does with its bf16 staging, and is why its
// timing curve has no notch at L=96.

#include "helix_common.h"

using namespace metal;

kernel void helix_pad_tail_f32(
        constant HelixScanParams& p     [[buffer(0)]],
        device const float*       Bt    [[buffer(5)]],
        device const float*       Ct    [[buffer(6)]],
        device       float*       Btail [[buffer(15)]],
        device       float*       Ctail [[buffer(16)]],
        uint3 gid [[thread_position_in_grid]]) {

    if (p.tail_t0 < 0) return;                    // sequence is a whole number of chunks

    const int n = int(gid.x);                     // state
    const int t = int(gid.y);                     // token within the chunk
    const int s = int(gid.z);                     // sequence
    if (n >= p.d_state || t >= HELIX_CS) return;

    const int src_t = p.tail_t0 + t;
    const bool live = (src_t < p.n_tok);

    for (int g = 0; g < p.n_group; ++g) {
        const int dst = s * p.tail_seq + t * p.tail_stride + g * p.d_state + n;
        Btail[dst] = live
            ? Bt[s * p.seq_b + src_t * p.s_b[0] + g * p.s_b[1] + n * p.s_b[2]]
            : 0.0f;
        Ctail[dst] = live
            ? Ct[s * p.seq_c + src_t * p.s_c[0] + g * p.s_c[1] + n * p.s_c[2]]
            : 0.0f;
    }
}
