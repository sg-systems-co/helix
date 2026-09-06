// helix.h — HELIX public C ABI.
//
// HELIX is a chunked-associative-scan (SSD) kernel for State-Space Models on
// Apple Silicon. It is deliberately a *pure encoder*: it appends compute
// encoders to a command buffer the caller owns, and it never allocates. Both
// properties exist so it can drop into GGML's graph submission without adding
// a command buffer, a sync point, or an allocation to the host runtime.
//
// The ABI is plain C so that llama.cpp can link HELIX as a static library
// without dragging Objective-C++ into every translation unit.
//
// Tensor layouts follow GGML's ssm_scan convention (fastest dimension last in
// the comments below):
//
//   s0, s1 : (n_seq, n_head, head_dim, d_state)  f32   -- carried state
//   x      : (n_seq, n_tok,  n_head,   head_dim)       -- input
//   dt     : (n_seq, n_tok,  n_head)                   -- pre-bias timestep
//   A      : (n_head)  scalar-per-head, or (n_head, d_state)  diagonal
//   B, C   : (n_seq, n_tok,  n_group,  d_state)
//   y      : (n_seq, n_tok,  n_head,   head_dim)       -- output
//
// `dt` arrives with its bias already folded in; HELIX applies softplus. The
// `D` skip connection is the caller's business and is applied outside.

#ifndef HELIX_H
#define HELIX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HELIX_ABI_VERSION 1

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

typedef enum {
    HELIX_OK = 0,
    HELIX_ERR_UNSUPPORTED_SHAPE = 1,  // helix_scan_supported() would return false
    HELIX_ERR_NO_DEVICE         = 2,  // Metal unavailable, or metallib missing
    HELIX_ERR_SCRATCH_TOO_SMALL = 3,
    HELIX_ERR_INVALID_ARG       = 4,
    HELIX_ERR_PIPELINE          = 5,  // pipeline state creation failed
} helix_status;

typedef enum {
    HELIX_F32  = 0,
    HELIX_F16  = 1,
    HELIX_BF16 = 2,
} helix_dtype;

typedef enum {
    // A is one scalar per head. This is Mamba-2 / SSD, and is the only form
    // upstream ggml-metal accelerates.
    HELIX_A_SCALAR_PER_HEAD = 0,
    // A is diagonal, one value per (head, state) pair. Mamba-1 selective scan.
    HELIX_A_DIAG_PER_CHANNEL = 1,
} helix_a_form;

typedef enum {
    HELIX_BACKEND_AUTO  = 0,
    HELIX_BACKEND_SGMMA = 1,  // simdgroup_matrix     (Apple M1+ / A14+)
    HELIX_BACKEND_MPP   = 2,  // MetalPerformancePrimitives tensor ops (M5 NA)
    HELIX_BACKEND_REF   = 3,  // scalar Metal fallback, for differential testing
} helix_backend;

// ---------------------------------------------------------------------------
// Tensors
// ---------------------------------------------------------------------------

// A view into caller-owned device memory. `buf` is a bridge-cast
// id<MTLBuffer>; `offset` is in BYTES from the start of that buffer. HELIX
// never retains, releases, or frees either.
typedef struct {
    void*    buf;
    uint64_t offset;
} helix_tensor;

#define HELIX_TENSOR_NULL ((helix_tensor){ NULL, 0 })

static inline bool helix_tensor_is_null(helix_tensor t) { return t.buf == NULL; }

// ---------------------------------------------------------------------------
// Descriptor
// ---------------------------------------------------------------------------

// All strides are in ELEMENTS, not bytes, mirroring ggml's nb[]/ne[] split so
// the GGML bridge is arithmetic rather than a copy. Index order matches the
// layout comments at the top of this file.
typedef struct {
    // Logical shape.
    int32_t n_tok;     // L  -- tokens in this slice
    int32_t n_seq;     // number of independent sequences
    int32_t n_head;    // H
    int32_t head_dim;  // P  (ggml calls this d_inner)
    int32_t d_state;   // N
    int32_t n_group;   // G  -- must divide n_head

    // Element strides.
    int64_t s_x[3];    // (tok, head, chan)
    int64_t s_b[3];    // (tok, group, state)
    int64_t s_c[3];    // (tok, group, state)
    int64_t s_dt[2];   // (tok, head)
    int64_t s_s[3];    // (head, chan, state) within one sequence
    int64_t s_y[3];    // (tok, head, chan)
    int64_t s_a;       // stride between heads in A (1 if scalar-per-head)
    int64_t seq_x;     // per-sequence strides
    int64_t seq_b;
    int64_t seq_c;
    int64_t seq_dt;
    int64_t seq_s;
    int64_t seq_y;

    // Algorithm knobs.
    int32_t       chunk;          // 0 => library chooses (currently 64)
    helix_a_form  a_form;
    helix_dtype   io_dtype;       // dtype of x/B/C/y in device memory
    helix_dtype   compute_dtype;  // matrix-unit input precision
    helix_backend backend;        // HELIX_BACKEND_AUTO in production
    bool          multipass;      // 3-pass chunk-parallel decomposition

    // softplus(dt) clamp. Set dt_min=0, dt_max=INFINITY to disable (this is
    // what ggml does; lite-ssm clamps).
    float dt_min;
    float dt_max;
} helix_scan_desc;

// Fills every field with the documented default and derives the dense,
// contiguous strides implied by the shape. Callers with non-dense tensors
// overwrite the stride fields afterwards.
void helix_scan_desc_init(helix_scan_desc* d,
                          int32_t n_tok, int32_t n_seq, int32_t n_head,
                          int32_t head_dim, int32_t d_state, int32_t n_group);

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

typedef struct helix_ctx helix_ctx;

// `device` is a bridge-cast id<MTLDevice>, or NULL to use the system default.
// `metallib_path` is NULL to use the search order documented in library.mm.
helix_status helix_ctx_create(helix_ctx** out, void* device, const char* metallib_path);
void         helix_ctx_free(helix_ctx* ctx);

// Human-readable description of the last error on this context. Never NULL.
const char*  helix_ctx_last_error(const helix_ctx* ctx);

// Which backend AUTO would select for this descriptor on this device.
helix_backend helix_ctx_select_backend(helix_ctx* ctx, const helix_scan_desc* d);

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

// True if HELIX has a kernel for this descriptor. When false, the caller must
// fall back to its own implementation -- HELIX is never a regression.
bool helix_scan_supported(helix_ctx* ctx, const helix_scan_desc* d);

// Bytes of device scratch required. Zero for the single-pass path.
size_t helix_scan_scratch_size(helix_ctx* ctx, const helix_scan_desc* d);

// Appends the scan to `cmd_buffer` (a bridge-cast id<MTLCommandBuffer>). Does
// not commit, does not wait, does not allocate.
//
// `ids` is an optional (n_seq) int32 tensor of source-state indices, matching
// ggml's ssm_scan `ids` argument. Pass HELIX_TENSOR_NULL for the identity map.
// `scratch` may be HELIX_TENSOR_NULL when helix_scan_scratch_size() is 0.
helix_status helix_scan_encode(helix_ctx*             ctx,
                               void*                  cmd_buffer,
                               const helix_scan_desc* d,
                               helix_tensor           s0,
                               helix_tensor           x,
                               helix_tensor           dt,
                               helix_tensor           A,
                               helix_tensor           B,
                               helix_tensor           C,
                               helix_tensor           ids,
                               helix_tensor           y,
                               helix_tensor           s1,
                               helix_tensor           scratch);

// As above, but appends to a compute encoder the caller already has open
// (`encoder` is a bridge-cast id<MTLComputeCommandEncoder>). Metal forbids two
// open encoders on one command buffer, so any host that is mid-graph -- ggml
// included -- must use this form rather than helix_scan_encode.
//
// HELIX reads the encoder's own dispatchType and inserts barriers between its
// passes when it is MTLDispatchTypeConcurrent, so the caller does not have to
// know or care. Bindings 0-14 are clobbered; the caller must rebind whatever it
// needs afterwards, which every ggml op already does.
helix_status helix_scan_encode_into(helix_ctx*             ctx,
                                    void*                  encoder,
                                    const helix_scan_desc* d,
                                    helix_tensor           s0,
                                    helix_tensor           x,
                                    helix_tensor           dt,
                                    helix_tensor           A,
                                    helix_tensor           B,
                                    helix_tensor           C,
                                    helix_tensor           ids,
                                    helix_tensor           y,
                                    helix_tensor           s1,
                                    helix_tensor           scratch);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // HELIX_H
