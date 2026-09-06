// helix_params.h -- the ONLY struct shared between the C++ runtime and MSL.
// Included from both, so it must stay plain C: no namespaces, no templates,
// no address-space qualifiers.
//
// Strides are element counts (not bytes), int32. The host validates that every
// derived index fits in int32 before dispatch -- 64-bit index arithmetic is
// materially slower on Apple GPUs and the largest realistic workload
// (n_seq*L*H*P) is far below 2^31.

#ifndef HELIX_PARAMS_H
#define HELIX_PARAMS_H

// Chunk size. 64 is the largest multiple of the 8x8 simdgroup tile whose
// working set fits the 32 KiB threadgroup budget; see the layout table in
// helix_scan_sgmma.metal.
#define HELIX_CS  64
// Head-dim slab width. head_dim > HELIX_HD is split across threadgroups in
// grid.x, which is what lets HELIX cover head_dim=128 where ggml falls back.
#define HELIX_HD  64
// Simdgroups per threadgroup.
#define HELIX_NSG 4
#define HELIX_SIMD_WIDTH 32
#define HELIX_TC 8   // simdgroup matrix tile edge
// Compile-time cap on chunks per super-chunk group, so the inter-chunk scan can
// stage its per-chunk decays in a fixed threadgroup array. The host clamps n_cg
// to this.
#define HELIX_MAX_CG 256
// k extent for the MPP matmuls whose reduction runs over d_state. Compile-time
// rather than dynamic_extent so the descriptor is a constant expression; the
// runtime only selects the MPP path when d_state matches.
#define HELIX_MPP_K 128

struct HelixScanParams {
    int n_tok;
    int n_head;
    int head_dim;
    int d_state;
    int n_group;
    int heads_per_group;
    int n_pslab;        // ceil(head_dim / HELIX_HD)

    // Element strides, matching helix_scan_desc.
    int s_x[3],  seq_x;
    int s_b[3],  seq_b;
    int s_c[3],  seq_c;
    int s_dt[2], seq_dt;
    int s_s[3],  seq_s;
    int s_y[3],  seq_y;
    int s_a;

    int   a_diagonal;   // 0 = scalar per head, 1 = diagonal per (head, state)
    int   has_ids;
    float dt_min;
    float dt_max;

    // --- multipass only -------------------------------------------------
    // The 3-pass decomposition runs over one "super-chunk" group at a time:
    // a serial host loop over groups bounds the scratch buffer, while the
    // chunks inside a group run fully in parallel. See §2.4 of the plan.
    int chunk_base;   // global index of this group's first chunk
    int n_cg;         // valid chunks in this group (<= HELIX_MAX_CG)
    int first_group;  // 1 => the carried state comes from s0, else from s1

    // bf16 operand staging, MPP path only. MPP reaches the M5 Neural
    // Accelerators only when BOTH operands are bf16 -- a bf16 x f32 matmul
    // measures 15.6 TFLOP/s against 65.7 for bf16 x bf16 -- so B, C and the
    // per-chunk initial state all need dense bf16 copies.
    //   Bbf/Cbf  [seq][tok][group][state]
    //   Sbf      [seq][cg][head][state][chan]   (shares sc_cg / sc_seq)
    int bf_tok, bf_seq;
    int n_tok_pad;   // n_tok rounded up to a whole chunk, for the staging arrays

    // Zero-padded copy of the final ragged chunk's B and C rows.
    //
    // The MMA path loads B and C in 8-row tiles, so a chunk with fewer than
    // HELIX_CS live tokens would read past the end of the sequence. Rather than
    // branch to a scalar tail -- which measured 6x slower than upstream at
    // L=96 -- the ragged chunk reads from here instead: rows [0, tlen) copied,
    // rows [tlen, HELIX_CS) zeroed. Zero rows contribute nothing through the
    // GEMMs, so the same code path handles both cases.
    //
    // Layout [seq][tok_in_chunk][group][state], dense. One chunk per sequence,
    // so this is HELIX_CS * n_group * d_state floats each for B and C --
    // 256 KiB at the Mamba-2 shape, against 67 MB to pad the whole sequence.
    int tail_t0;       // first token of the ragged chunk, or -1 if none
    int tail_stride;   // n_group * d_state
    int tail_seq;      // HELIX_CS * n_group * d_state

    // Scratch strides, in elements.
    //   dS      [seq][cg][head][state][chan]   -- state OUTER, chan contiguous
    //   logdec  [seq][cg][head]
    //
    // The state-major layout is what lets both GEMMs that touch dS run without
    // a transposed operand: Pass A's G4 produces (state x chan) tiles directly,
    // and Pass C's G3 consumes them with a plain load.
    int sc_cg,  sc_seq;
    int ld_cg,  ld_seq;
};

#endif  // HELIX_PARAMS_H
