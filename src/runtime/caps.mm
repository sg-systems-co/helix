// caps.mm -- backend selection, shape support, scratch sizing.
//
// helix_scan_supported() is the contract with the host runtime: when it says
// false the caller keeps its own implementation, so HELIX can never be a
// regression. Everything it rejects is something a later milestone adds.

#include <cstdint>
#include <cstdlib>
#include <limits>

#include "helix_internal.h"

namespace {

// M1-M4 and iPhone A14+. fp32 simdgroup_matrix: measured at 16.12 TFLOP/s,
// against 10.88 for the same path at bf16, so fp32 is not a compromise here --
// it is the fast option.
constexpr HelixMatmulTraits kSgmmaTraits = {
    HELIX_BACKEND_SGMMA, "sgmma",
    "helix_scan_sgmma_f32",
    "helix_chunk_state_f32",
    "helix_state_scan_f32",
    "helix_chunk_out_f32",
    /*needs_bf16_staging=*/false,
};

// M5 (Apple10). Pass C runs on the Neural Accelerators via MPP matmul2d in
// bf16: 65.7 TFLOP/s, 4.19x the simdgroup path. Passes A and B stay on the
// portable kernels -- A's GEMM already runs at fp32 simdgroup rate and B is a
// memory-bound scan with no matmul in it.
constexpr HelixMatmulTraits kMppTraits = {
    HELIX_BACKEND_MPP, "mpp",
    "helix_scan_sgmma_f32",
    "helix_chunk_state_f32",
    "helix_state_scan_f32",
    "helix_chunk_out_mpp",
    /*needs_bf16_staging=*/true,
};

// The MPP Pass C is compiled against fixed tile extents, so the descriptor has
// to match them exactly.
bool mpp_shape_ok(const helix_scan_desc& d) {
    if (!d.multipass) return false;              // single-pass has no Pass C
    if (d.d_state != HELIX_MPP_K) return false;  // k extent is compile-time
    // The state tensor spans a full HELIX_HD-wide slab; a partial trailing slab
    // would read past the end of each row.
    if (d.head_dim % HELIX_HD != 0) return false;
    return true;
}

}  // namespace

namespace helix::rt {

const HelixMatmulTraits& select_traits(helix_ctx* ctx, const helix_scan_desc& d) {
    if (!ctx) return kSgmmaTraits;

    // Test hook: HELIX_FORCE_BACKEND=sgmma exercises the M1-M4 path on M5
    // hardware, which is the only way to regression-test the fallback without a
    // second machine.
    if (const char* f = std::getenv("HELIX_FORCE_BACKEND"))
        if (f[0] == 's') return kSgmmaTraits;

    const bool want_mpp = (d.backend == HELIX_BACKEND_AUTO || d.backend == HELIX_BACKEND_MPP);

    if (want_mpp && ctx->has_neural_accel && ctx->mpp_library && mpp_shape_ok(d))
        return kMppTraits;

    return kSgmmaTraits;
}

}  // namespace helix::rt

extern "C" helix_backend helix_ctx_select_backend(helix_ctx* ctx, const helix_scan_desc* d) {
    if (!ctx || !d) return HELIX_BACKEND_REF;
    if (!ctx->has_simdgroup_mm) return HELIX_BACKEND_REF;
    if (d->backend == HELIX_BACKEND_SGMMA) return HELIX_BACKEND_SGMMA;
    return helix::rt::select_traits(ctx, *d).backend;
}

extern "C" bool helix_scan_supported(helix_ctx* ctx, const helix_scan_desc* d) {
    if (!ctx || !d) return false;
    if (!ctx->has_simdgroup_mm) return false;

    if (d->n_tok <= 0 || d->n_seq <= 0 || d->n_head <= 0 ||
        d->head_dim <= 0 || d->d_state <= 0 || d->n_group <= 0) return false;

    if (d->n_head % d->n_group != 0) return false;

    // The reassociation into a dense M matrix requires the decay to be one
    // scalar per (head, timestep). Diagonal-A is Week 4.
    if (d->a_form != HELIX_A_SCALAR_PER_HEAD) return false;

    // fp16/bf16 storage is Week 3.
    if (d->io_dtype != HELIX_F32 || d->compute_dtype != HELIX_F32) return false;

    // The 8x8 simdgroup tiles walk the state dimension in steps of 8.
    if (d->d_state % HELIX_TC != 0) return false;

    // head_dim is split into HELIX_HD-wide slabs across grid.x, so any multiple
    // of the tile edge works -- this is what covers head_dim=128, where ggml
    // falls back to its scalar kernel entirely.
    if (d->head_dim % HELIX_TC != 0) return false;

    // The MMA loads of B, C and the state address memory directly, so the
    // innermost dimension of each must be contiguous.
    if (d->s_b[2] != 1 || d->s_c[2] != 1 || d->s_s[2] != 1) return false;
    if (d->s_s[1] != d->d_state) return false;

    if (d->chunk != 0 && d->chunk != HELIX_CS) return false;

    // An explicit MPP request must actually be serviceable; AUTO silently falls
    // back to simdgroup_matrix and is always fine.
    if (d->backend == HELIX_BACKEND_MPP &&
        helix::rt::select_traits(ctx, *d).backend != HELIX_BACKEND_MPP)
        return false;

    HelixScanParams unused;
    return helix::rt::build_params(ctx, *d, &unused);
}

extern "C" size_t helix_scan_scratch_size(helix_ctx* ctx, const helix_scan_desc* d) {
    if (!d) return 0;
    // The single-pass kernel carries state through the caller's s1 buffer and
    // needs nothing else.
    if (!d->multipass) return 0;
    const bool staging = helix::rt::select_traits(ctx, *d).needs_bf16_staging;
    return helix::rt::plan_multipass(*d, staging).total_bytes;
}

namespace helix::rt {

namespace {

// Live scratch budget. Larger groups mean more chunks in flight and better
// occupancy, but scratch grows linearly with the group size. 128 MB is enough
// to put every chunk of a 4K prefill in flight at the Mamba-2 shape while
// staying well inside a phone-sized allocation.
size_t scratch_budget_bytes() {
    if (const char* env = std::getenv("HELIX_SCRATCH_MB")) {
        const long mb = std::strtol(env, nullptr, 10);
        if (mb > 0) return (size_t)mb << 20;
    }
    return (size_t)128 << 20;
}

}  // namespace

MultipassPlan plan_multipass(const helix_scan_desc& d, bool staging) {
    MultipassPlan pl;

    const int cs = (d.chunk > 0) ? d.chunk : HELIX_CS;
    pl.n_chunks = (d.n_tok + cs - 1) / cs;
    if (pl.n_chunks <= 0) return pl;

    auto align256 = [](size_t v) { return (v + 255) & ~(size_t)255; };

    // B and C staging scales with sequence length, not with the group size, so
    // it comes off the budget before the per-chunk arithmetic.
    // Padded to a whole number of chunks. Pass C builds a CS-row tensor over
    // every chunk including the ragged last one, so the staging array must have
    // those rows present (zero-filled) rather than relying on MPP edge
    // semantics to suppress an out-of-bounds read.
    const int    cs_pad   = (d.n_tok + cs - 1) / cs * cs;
    const size_t bc_elems = staging
        ? (size_t)d.n_seq * cs_pad * d.n_group * d.d_state : 0;
    const size_t bc_bytes  = align256(bc_elems * sizeof(uint16_t)) * 2;

    size_t budget = scratch_budget_bytes();
    budget = (budget > bc_bytes) ? budget - bc_bytes : 0;

    // Per chunk: the f32 dS the scan consumes, plus (on the MPP path) the bf16
    // copy of the initial state that Pass C reads.
    const size_t state_elems = (size_t)d.n_seq * d.n_head * d.head_dim * d.d_state;
    const size_t per_chunk   = state_elems * (sizeof(float) + (staging ? sizeof(uint16_t) : 0));

    size_t cap = per_chunk ? budget / per_chunk : 1;
    if (cap < 1) cap = 1;                       // always admit at least one chunk
    if (cap > (size_t)HELIX_MAX_CG) cap = HELIX_MAX_CG;
    if (cap > (size_t)pl.n_chunks)  cap = pl.n_chunks;

    pl.chunks_per_group = (int)cap;
    pl.n_groups = (pl.n_chunks + pl.chunks_per_group - 1) / pl.chunks_per_group;

    // One zero-padded chunk of B and C per sequence, for the ragged tail. Tiny
    // next to dS -- 256 KiB at the Mamba-2 shape -- and it is what lets every
    // chunk take the same GEMM path.
    pl.tail_elems = (size_t)d.n_seq * HELIX_CS * d.n_group * d.d_state;

    pl.ds_elems  = state_elems * pl.chunks_per_group;
    pl.ld_elems  = (size_t)d.n_seq * pl.chunks_per_group * d.n_head;
    pl.bc_elems  = bc_elems;
    pl.sbf_elems = staging ? pl.ds_elems : 0;

    // Every region is 256-byte aligned so each can be bound as its own buffer
    // view without a misaligned offset.
    size_t off = 0;
    off += align256(pl.ds_elems * sizeof(float));
    pl.off_ld = off;
    off += align256(pl.ld_elems * sizeof(float));
    if (staging) {
        pl.off_bbf = off;  off += align256(pl.bc_elems  * sizeof(uint16_t));
        pl.off_cbf = off;  off += align256(pl.bc_elems  * sizeof(uint16_t));
        pl.off_sbf = off;  off += align256(pl.sbf_elems * sizeof(uint16_t));
    }
    pl.off_btail = off;  off += align256(pl.tail_elems * sizeof(float));
    pl.off_ctail = off;  off += align256(pl.tail_elems * sizeof(float));
    pl.total_bytes = off;
    return pl;
}

bool build_params(helix_ctx* ctx, const helix_scan_desc& d, HelixScanParams* o) {
    // Every stride below is stored as int32 in HelixScanParams; 64-bit index
    // math is materially slower on Apple GPUs and no realistic workload needs
    // it. Verify rather than assume.
    const int64_t kMax = std::numeric_limits<int32_t>::max();
    auto fits = [&](int64_t v) { return v >= -kMax && v <= kMax; };

    const int64_t biggest = (int64_t)d.n_seq * d.seq_x +
                            (int64_t)d.n_tok * d.s_x[0] +
                            (int64_t)d.n_head * d.head_dim;
    if (!fits(biggest) || !fits((int64_t)d.n_seq * d.seq_s) ||
        !fits((int64_t)d.n_seq * d.seq_b) || !fits((int64_t)d.n_seq * d.seq_y)) {
        if (ctx) ctx->last_error = "tensor indices exceed int32 stride range";
        return false;
    }

    o->n_tok           = d.n_tok;
    o->n_head          = d.n_head;
    o->head_dim        = d.head_dim;
    o->d_state         = d.d_state;
    o->n_group         = d.n_group;
    o->heads_per_group = d.n_head / d.n_group;
    o->n_pslab         = (d.head_dim + HELIX_HD - 1) / HELIX_HD;

    for (int i = 0; i < 3; ++i) {
        o->s_x[i] = (int)d.s_x[i];
        o->s_b[i] = (int)d.s_b[i];
        o->s_c[i] = (int)d.s_c[i];
        o->s_s[i] = (int)d.s_s[i];
        o->s_y[i] = (int)d.s_y[i];
    }
    o->s_dt[0] = (int)d.s_dt[0];
    o->s_dt[1] = (int)d.s_dt[1];

    o->seq_x  = (int)d.seq_x;
    o->seq_b  = (int)d.seq_b;
    o->seq_c  = (int)d.seq_c;
    o->seq_dt = (int)d.seq_dt;
    o->seq_s  = (int)d.seq_s;
    o->seq_y  = (int)d.seq_y;
    o->s_a    = (int)d.s_a;

    {
        const int cs     = (d.chunk > 0) ? d.chunk : HELIX_CS;
        const int cs_pad = (d.n_tok + cs - 1) / cs * cs;
        o->bf_tok    = (int)((int64_t)d.n_group * d.d_state);
        o->bf_seq    = (int)((int64_t)cs_pad * d.n_group * d.d_state);
        o->n_tok_pad = cs_pad;

        // -1 when the sequence is a whole number of chunks and no padding is
        // needed; otherwise the first token of the ragged final chunk.
        o->tail_t0     = (d.n_tok % cs == 0) ? -1 : (d.n_tok / cs) * cs;
        o->tail_stride = (int)((int64_t)d.n_group * d.d_state);
        o->tail_seq    = (int)((int64_t)cs * d.n_group * d.d_state);
    }

    o->a_diagonal = (d.a_form == HELIX_A_DIAG_PER_CHANNEL) ? 1 : 0;
    o->has_ids    = 0;  // set by the encoder, which knows whether ids was passed
    o->dt_min     = d.dt_min;
    o->dt_max     = d.dt_max;
    return true;
}

}  // namespace helix::rt
