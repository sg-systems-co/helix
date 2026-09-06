// ggml-helix.h -- HELIX interception point for ggml-metal's SSM_SCAN.
//
// One entry point, called from a single #ifdef inside ggml_metal_op_ssm_scan.
// Returns false for anything HELIX cannot serve, and the caller then runs its
// own dispatch unchanged -- HELIX is never a regression.

#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_metal_encoder;

// `enc` is ggml's encoder for the current graph segment. Returns true if HELIX
// encoded the op, false if the caller must handle it.
bool ggml_metal_op_ssm_scan_helix(struct ggml_metal_encoder * enc, const struct ggml_tensor * op);

#ifdef __cplusplus
}
#endif
