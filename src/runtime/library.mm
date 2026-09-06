// library.mm -- metallib resolution and the pipeline cache.

#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import <Metal/Metal.h>

#include <mach-o/dyld.h>
#include <sys/stat.h>

#include <cstdlib>
#include <vector>

#include "helix_internal.h"

namespace {

bool file_exists(const std::string& p) {
    struct stat st;
    return !p.empty() && ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string executable_dir() {
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buf(size + 1, 0);
    if (_NSGetExecutablePath(buf.data(), &size) != 0) return {};
    std::string path(buf.data());
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? std::string{} : path.substr(0, slash);
}

}  // namespace

#ifdef HELIX_HAVE_EMBEDDED_METALLIB
extern "C" {
extern const unsigned char helix_metallib_data[];
extern const size_t        helix_metallib_data_len;
#ifdef HELIX_HAVE_EMBEDDED_MPP_METALLIB
extern const unsigned char helix_mpp_metallib_data[];
extern const size_t        helix_mpp_metallib_data_len;
#endif
}
#endif

namespace helix::rt {

// Builds a library from bytes compiled into this binary. Preferred over the
// on-disk path so that a shipped libhelix.a needs no loose files beside it; an
// explicit path or $HELIX_METALLIB still wins, which is what makes shader
// iteration practical without a rebuild.
id<MTLLibrary> embedded_library(id<MTLDevice> dev, bool mpp, std::string* err) {
    (void)dev; (void)mpp; (void)err;
#ifdef HELIX_HAVE_EMBEDDED_METALLIB
    const void*  bytes = nullptr;
    size_t       len   = 0;
    if (!mpp) {
        bytes = helix_metallib_data;  len = helix_metallib_data_len;
    } else {
#ifdef HELIX_HAVE_EMBEDDED_MPP_METALLIB
        bytes = helix_mpp_metallib_data;  len = helix_mpp_metallib_data_len;
#else
        return nil;
#endif
    }
    if (!bytes || len == 0) return nil;

    // dispatch_data_create with DISPATCH_DATA_DESTRUCTOR_DEFAULT copies, which
    // is wrong for a multi-megabyte constant that already lives in __TEXT. The
    // destructor below is a no-op because the bytes are static.
    dispatch_data_t dd = dispatch_data_create(bytes, len, nullptr,
                                              ^{ /* static storage, nothing to free */ });
    NSError* nserr = nil;
    id<MTLLibrary> lib = [dev newLibraryWithData:dd error:&nserr];
    if (!lib && err)
        *err = nserr ? nserr.localizedDescription.UTF8String : "newLibraryWithData failed";
    return lib;
#else
    return nil;
#endif
}

std::string resolve_metallib(const char* explicit_path) {
    if (explicit_path && file_exists(explicit_path)) return explicit_path;

    if (const char* env = std::getenv("HELIX_METALLIB"); env && file_exists(env))
        return env;

    // The compile-time build-tree path is only consulted when nothing was
    // embedded. Leaving it ahead of the embedded copy would mean a shipped
    // binary silently depends on a build directory that happens to still exist
    // on the developer's machine -- and that the embedding was never exercised.
#if defined(HELIX_METALLIB_PATH) && !defined(HELIX_HAVE_EMBEDDED_METALLIB)
    if (file_exists(HELIX_METALLIB_PATH)) return HELIX_METALLIB_PATH;
#endif

    const std::string local = executable_dir() + "/helix.metallib";
    if (file_exists(local)) return local;

    return {};
}

// The MPP library is optional in every sense: it may not have been built (older
// toolchain), and it must not be loaded at all on a pre-Apple10 device.
std::string resolve_mpp_metallib() {
    if (const char* env = std::getenv("HELIX_MPP_METALLIB"); env && file_exists(env))
        return env;

#if defined(HELIX_MPP_METALLIB_PATH) && !defined(HELIX_HAVE_EMBEDDED_MPP_METALLIB)
    if (file_exists(HELIX_MPP_METALLIB_PATH)) return HELIX_MPP_METALLIB_PATH;
#endif

    const std::string local = executable_dir() + "/helix_mpp.metallib";
    if (file_exists(local)) return local;

    return {};
}

id<MTLComputePipelineState> pipeline(helix_ctx* ctx, const std::string& name) {
    if (auto it = ctx->pipelines.find(name); it != ctx->pipelines.end()) return it->second;

    // The MPP library is searched first so an M5 build can override a kernel
    // name; on any other device it is nil and this is a single lookup.
    id<MTLFunction> fn = nil;
    if (ctx->mpp_library) fn = [ctx->mpp_library newFunctionWithName:@(name.c_str())];
    if (!fn)              fn = [ctx->library     newFunctionWithName:@(name.c_str())];
    if (!fn) {
        ctx->last_error = "kernel not found in metallib: " + name;
        return nil;
    }

    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        [ctx->device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        ctx->last_error = "pipeline creation failed for " + name + ": " +
                          (err ? err.localizedDescription.UTF8String : "unknown");
        return nil;
    }

    ctx->pipelines.emplace(name, pso);
    return pso;
}

}  // namespace helix::rt
