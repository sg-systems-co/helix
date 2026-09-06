// device.mm -- context lifetime and capability probing.

#import <Metal/Metal.h>

#include "helix_internal.h"

extern "C" helix_status helix_ctx_create(helix_ctx** out, void* device,
                                         const char* metallib_path) {
    if (!out) return HELIX_ERR_INVALID_ARG;
    *out = nullptr;

    auto* ctx = new helix_ctx();

    ctx->device = device ? (__bridge id<MTLDevice>)device : MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        delete ctx;
        return HELIX_ERR_NO_DEVICE;
    }

    // Apple7 is the first family with simdgroup_matrix (M1, A14). Apple10 is
    // M5, the first with per-core Neural Accelerators reachable through
    // MetalPerformancePrimitives. The MPP path is gated on this but not yet
    // implemented, so selection still falls through to SGMMA.
    ctx->has_simdgroup_mm      = [ctx->device supportsFamily:MTLGPUFamilyApple7];
    ctx->has_neural_accel      = [ctx->device supportsFamily:MTLGPUFamilyApple10];
    ctx->max_threadgroup_bytes = (size_t)ctx->device.maxThreadgroupMemoryLength;

    // An explicit path or $HELIX_METALLIB wins, so shader iteration does not
    // need a rebuild of the embedded copy. Otherwise use the bytes linked into
    // this binary, and only fall back to disk if nothing was embedded.
    ctx->metallib_path = helix::rt::resolve_metallib(metallib_path);

    if (ctx->metallib_path.empty()) {
        std::string emb_err;
        ctx->library = helix::rt::embedded_library(ctx->device, /*mpp=*/false, &emb_err);
        if (ctx->library) ctx->metallib_path = "<embedded>";
        else if (!emb_err.empty()) ctx->last_error = emb_err;
    }

    if (!ctx->library) {
        if (ctx->metallib_path.empty() || ctx->metallib_path == "<embedded>") {
            ctx->last_error = "no metallib: none embedded and none found on disk"
                              " (set $HELIX_METALLIB)";
            delete ctx;
            return HELIX_ERR_NO_DEVICE;
        }
        NSError* err = nil;
        NSURL*   url = [NSURL fileURLWithPath:@(ctx->metallib_path.c_str())];
        ctx->library = [ctx->device newLibraryWithURL:url error:&err];
        if (!ctx->library) {
            ctx->last_error = std::string("failed to load metallib: ") +
                              (err ? err.localizedDescription.UTF8String : "unknown");
            delete ctx;
            return HELIX_ERR_NO_DEVICE;
        }
    }

    // Load the MPP library only on hardware that has the Neural Accelerators.
    // Failure here is never fatal: select_traits() falls back to the portable
    // simdgroup_matrix kernels, which is also the M1-M4 path.
    if (ctx->has_neural_accel) {
        ctx->mpp_metallib_path = helix::rt::resolve_mpp_metallib();
        if (!ctx->mpp_metallib_path.empty()) {
            NSError* mpp_err = nil;
            NSURL* mpp_url = [NSURL fileURLWithPath:@(ctx->mpp_metallib_path.c_str())];
            ctx->mpp_library = [ctx->device newLibraryWithURL:mpp_url error:&mpp_err];
            if (!ctx->mpp_library)
                ctx->last_error = std::string("MPP metallib present but failed to load: ") +
                                  (mpp_err ? mpp_err.localizedDescription.UTF8String : "unknown");
        } else {
            std::string emb_err;
            ctx->mpp_library = helix::rt::embedded_library(ctx->device, /*mpp=*/true, &emb_err);
            if (ctx->mpp_library) ctx->mpp_metallib_path = "<embedded>";
        }
    }

    *out = ctx;
    return HELIX_OK;
}

extern "C" void helix_ctx_free(helix_ctx* ctx) { delete ctx; }

extern "C" const char* helix_ctx_last_error(const helix_ctx* ctx) {
    return ctx ? ctx->last_error.c_str() : "null context";
}
