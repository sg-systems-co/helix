#pragma once
// helix_internal.h -- ObjC++ internals. Not installed, not part of the ABI.

#import <Metal/Metal.h>

#include <string>
#include <unordered_map>

#include "helix/helix.h"
#include "../kernels/helix_params.h"

struct helix_ctx {
    id<MTLDevice>  device      = nil;
    id<MTLLibrary> library     = nil;   // portable, metal3.2
    id<MTLLibrary> mpp_library = nil;   // MPP tensor ops, metal4.0, Apple10 only
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    // Capabilities, probed once at context creation.
    bool   has_simdgroup_mm      = false;  // Apple7 (M1 / A14) and later
    bool   has_neural_accel      = false;  // Apple10 (M5): MPP tensor ops
    size_t max_threadgroup_bytes = 0;

    std::string metallib_path;
    std::string mpp_metallib_path;
    std::string last_error;
};

// ---------------------------------------------------------------------------
// Matmul backends
// ---------------------------------------------------------------------------
//
// Which matrix path a kernel takes is a property of the device, not of the
// algorithm, so it is resolved once per dispatch and the kernels are selected by
// name. The two backends are not interchangeable on precision: see
// docs/PRECISION.md. In short, on M5 the Neural Accelerators are reachable only
// through MPP and only with bf16 operands, while on M1-M4 simdgroup_matrix at
// fp32 is both the fastest and the only option.
struct HelixMatmulTraits {
    helix_backend backend           = HELIX_BACKEND_SGMMA;
    const char*   name              = "sgmma";
    const char*   k_single          = nullptr;  // 1-pass kernel
    const char*   k_chunk_state     = nullptr;  // Pass A
    const char*   k_state_scan      = nullptr;  // Pass B
    const char*   k_chunk_out       = nullptr;  // Pass C
    // MPP needs dense bf16 copies of B, C and the per-chunk initial state,
    // because a bf16 x f32 matmul does not reach the Neural Accelerators.
    bool          needs_bf16_staging = false;
};

namespace helix::rt {

// Never fails: falls back to SgmmaTraits when the MPP path is unavailable or
// the shape is outside what it supports.
const HelixMatmulTraits& select_traits(helix_ctx* ctx, const helix_scan_desc& d);

}  // namespace helix::rt

namespace helix::rt {

// Returns nil and sets ctx->last_error on failure. Pipelines are cached.
id<MTLComputePipelineState> pipeline(helix_ctx* ctx, const std::string& name);

// Resolves helix.metallib. Search order:
//   1. explicit argument
//   2. $HELIX_METALLIB
//   3. HELIX_METALLIB_PATH baked in by CMake
//   4. <directory of the running executable>/helix.metallib
std::string resolve_metallib(const char* explicit_path);

// Optional. Empty when the MPP library was not built or is not present.
std::string resolve_mpp_metallib();

// Library built from metallib bytes linked into this binary. Returns nil when
// nothing was embedded at build time.
id<MTLLibrary> embedded_library(id<MTLDevice> dev, bool mpp, std::string* err);

// Fills the MSL-side params from the public descriptor. Returns false (and sets
// last_error) if any index would overflow the int32 stride fields.
bool build_params(helix_ctx* ctx, const helix_scan_desc& d, HelixScanParams* out);

// How the 3-pass decomposition is tiled over the sequence.
//
// The whole point of the decomposition is to expose n_chunks x more
// threadgroups, but materializing every chunk's dS at once costs
// n_chunks * n_seq * H * P * N floats -- 268 MB for an 8K prefill. So the host
// walks groups of at most `chunks_per_group` chunks serially, carrying state
// between groups through s1, and only that many chunks' worth of scratch is
// ever live.
//
// This is a pure function of the descriptor: helix_scan_scratch_size() and the
// encoder must agree on it exactly, or the encoder writes past the buffer the
// caller sized.
struct MultipassPlan {
    int    n_chunks         = 0;
    int    chunks_per_group = 0;
    int    n_groups         = 0;
    size_t ds_elems         = 0;   // dS region            (f32)
    size_t ld_elems         = 0;   // per-chunk log-decay  (f32)
    size_t bc_elems         = 0;   // B and C staging, each (bf16, MPP only)
    size_t sbf_elems        = 0;   // initial-state staging (bf16, MPP only)
    size_t tail_elems       = 0;   // zero-padded ragged chunk, each of B and C (f32)
    // Byte offsets into the caller's scratch buffer.
    size_t off_ld = 0, off_bbf = 0, off_cbf = 0, off_sbf = 0;
    size_t off_btail = 0, off_ctail = 0;
    size_t total_bytes      = 0;
};

// `staging` selects whether the bf16 regions are included. Must match what the
// encoder does, or the encoder writes past the buffer the caller sized.
MultipassPlan plan_multipass(const helix_scan_desc& d, bool staging);

}  // namespace helix::rt
