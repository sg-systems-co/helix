// test_metal_parity.mm -- Milestone 2 gate.
//
// Runs the Metal kernel against the fp64 sequential reference across the same
// shape matrix the algebra test uses, including every sequence-length edge case
// and the head_dim=128 shape where ggml-metal falls back to its scalar kernel.

#import <Metal/Metal.h>

#include <cstdio>
#include <string>
#include <vector>
#include <cstdlib>

#include "harness.hpp"
#include "helix/helix.h"
#include "helix/reference.hpp"

using namespace helix;
using namespace helix::test;

namespace {

// Two calibrated tolerance pairs, selected by the backend that actually ran.
//
// fp32: sweeping this shape matrix puts the failure edge at rel~2.6e-6 /
// abs~2.6e-7 (accumulated fp32 error of a 64-token chunk over a 128-deep state
// reduction). These sit ~2x above it.
//
// bf16 (MPP): the GEMM operands carry an 8-bit mantissa, so the floor is
// 2^-8 ~ 3.9e-3 per rounded value regardless of how the reduction is done.
// Holding that path to the fp32 bound would be a category error, but the bound
// still has to be tight enough to catch a real bug. Swept the same way as the
// fp32 pair: the failure edge sits at rel~7e-3 / abs~1.4e-3, so these are ~2x
// above it.
constexpr double kRelTolF32  = 5e-6;
constexpr double kAbsTolF32  = 5e-7;
constexpr double kRelTolBf16 = 1.5e-2;
constexpr double kAbsTolBf16 = 3e-3;

struct Gpu {
    id<MTLDevice>       device = nil;
    id<MTLCommandQueue> queue  = nil;
    helix_ctx*          ctx    = nullptr;

    bool init(std::string* err) {
        device = MTLCreateSystemDefaultDevice();
        if (!device) { *err = "no Metal device"; return false; }
        queue = [device newCommandQueue];
        const helix_status st =
            helix_ctx_create(&ctx, (__bridge void*)device, nullptr);
        if (st != HELIX_OK) {
            *err = "helix_ctx_create failed: " + std::to_string((int)st);
            return false;
        }
        return true;
    }

    id<MTLBuffer> upload(const void* p, size_t bytes) {
        if (bytes == 0) bytes = 4;
        return [device newBufferWithBytes:(p ? p : (const void*)"\0\0\0\0")
                                   length:bytes
                                  options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> alloc(size_t bytes) {
        return [device newBufferWithLength:(bytes ? bytes : 4)
                                   options:MTLResourceStorageModeShared];
    }
};

helix_tensor tensor_of(id<MTLBuffer> b) { return helix_tensor{(__bridge void*)b, 0}; }

struct Shape {
    int32_t L, n_seq, H, P, N, G;
    const char* label;
};

const Shape kShapes[] = {
    {   1, 1,  2,  64, 128, 1, "L=1 ragged only"                 },
    {   7, 1,  2,  64, 128, 1, "L=7 ragged only"                 },
    {  63, 1,  2,  64, 128, 1, "L=63 ragged only"                },
    {  64, 1,  2,  64, 128, 1, "L=64 one full chunk"             },
    {  65, 1,  2,  64, 128, 1, "L=65 full + 1 ragged"            },
    { 127, 1,  2,  64, 128, 1, "L=127 full + ragged"             },
    { 128, 1,  2,  64, 128, 1, "L=128 two full chunks"           },
    { 129, 1,  2,  64, 128, 1, "L=129 two full + ragged"         },
    {  96, 1,  4,  64, 128, 1, "L=96 full + 32 ragged"           },
    { 160, 1,  4,  64, 128, 1, "L=160 two full + 32 ragged"      },
    { 300, 1,  8,  64, 128, 2, "L=300 four full + 44 ragged"     },
    { 256, 1,  4,  64,  64, 1, "d_state=64"                      },
    { 256, 1,  4, 128, 128, 1, "head_dim=128 (ggml falls back)"   },
    { 256, 1,  4, 192, 128, 1, "head_dim=192 three slabs"        },
    { 256, 1,  8, 128, 128, 8, "n_group=8 grouped"               },
    { 256, 4,  4,  64, 128, 1, "n_seq=4 batched"                 },
    {1024, 1,  2,  64, 128, 1, "L=1024 sixteen chunks"           },
    {2048, 1,  8,  64, 128, 2, "L=2048 realistic prefill"        },
};

// The three dispatch shapes worth distinguishing. The tiny scratch budget is
// not a performance setting -- it forces plan_multipass() into many groups, so
// the state carried from one super-chunk group to the next is actually
// exercised. With a large budget an 8-chunk problem fits in a single group and
// that path never runs.
struct Mode {
    const char* label;
    bool        multipass;
    const char* scratch_mb;   // nullptr => leave the default
    Memory      mem;
};

// SHORT-memory inputs decay so hard (per-chunk decay ~1e-14) that the
// inter-chunk state carry is numerically irrelevant and bugs in it are
// invisible. The LONG rows are the ones that actually cover that path.
const Mode kModes[] = {
    {"1pass",         false, nullptr, Memory::SHORT},
    {"3pass",         true,  nullptr, Memory::SHORT},
    {"3pass/sm",      true,  "1",     Memory::SHORT},
    {"1pass/long",    false, nullptr, Memory::LONG },
    {"3pass/long",    true,  nullptr, Memory::LONG },
    {"3pass/sm/long", true,  "1",     Memory::LONG },
};

}  // namespace

int main() {
    Suite suite("test_metal_parity");

    Gpu gpu;
    std::string err;
    if (!gpu.init(&err)) {
        std::printf("SKIP: %s\n", err.c_str());
        return 0;  // no GPU in this environment is not a test failure
    }
    std::printf("device: %s  (simdgroup_mm=%d neural_accel=%d)\n",
                gpu.device.name.UTF8String,
                [gpu.device supportsFamily:MTLGPUFamilyApple7],
                [gpu.device supportsFamily:MTLGPUFamilyApple10]);

    for (const Mode& mode : kModes) {
      if (mode.scratch_mb) ::setenv("HELIX_SCRATCH_MB", mode.scratch_mb, 1);
      else                 ::unsetenv("HELIX_SCRATCH_MB");

      for (const Shape& sh : kShapes) {
        Problem p = Problem::make(sh.L, sh.n_seq, sh.H, sh.P, sh.N, sh.G, 0, 4242, mode.mem);
        p.d.multipass = mode.multipass;
        const std::string tag = std::string(mode.label) + " " + sh.label;

        if (!helix_scan_supported(gpu.ctx, &p.d)) {
            suite.check(false, "unsupported shape: " + tag);
            continue;
        }

        // Tolerance follows the backend that will actually run, not the mode
        // label: AUTO silently falls back to fp32 simdgroup_matrix for shapes
        // outside the MPP kernel's fixed tile extents.
        const bool is_mpp =
            helix_ctx_select_backend(gpu.ctx, &p.d) == HELIX_BACKEND_MPP;
        const double rel_tol = is_mpp ? kRelTolBf16 : kRelTolF32;
        const double abs_tol = is_mpp ? kAbsTolBf16 : kAbsTolF32;
        const std::string btag = tag + (is_mpp ? " [mpp]" : "");

        std::vector<float> y_ref(p.y_elems()), s_ref(p.s_elems());
        ref::scan(p.d, p.s0.data(), p.x.data(), p.dt.data(), p.A.data(),
                  p.B.data(), p.C.data(), p.ids.data(), y_ref.data(), s_ref.data());

        @autoreleasepool {
            id<MTLBuffer> b_s0 = gpu.upload(p.s0.data(), p.s0.size() * 4);
            id<MTLBuffer> b_x  = gpu.upload(p.x.data(),  p.x.size()  * 4);
            id<MTLBuffer> b_dt = gpu.upload(p.dt.data(), p.dt.size() * 4);
            id<MTLBuffer> b_A  = gpu.upload(p.A.data(),  p.A.size()  * 4);
            id<MTLBuffer> b_B  = gpu.upload(p.B.data(),  p.B.size()  * 4);
            id<MTLBuffer> b_C  = gpu.upload(p.C.data(),  p.C.size()  * 4);
            id<MTLBuffer> b_y  = gpu.alloc(p.y_elems() * 4);
            id<MTLBuffer> b_s1 = gpu.alloc(p.s_elems() * 4);

            const size_t scratch_bytes = helix_scan_scratch_size(gpu.ctx, &p.d);
            id<MTLBuffer> b_sc = scratch_bytes ? gpu.alloc(scratch_bytes) : nil;
            const helix_tensor t_sc =
                b_sc ? tensor_of(b_sc) : HELIX_TENSOR_NULL;

            id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
            const helix_status st = helix_scan_encode(
                gpu.ctx, (__bridge void*)cb, &p.d,
                tensor_of(b_s0), tensor_of(b_x), tensor_of(b_dt), tensor_of(b_A),
                tensor_of(b_B), tensor_of(b_C), HELIX_TENSOR_NULL,
                tensor_of(b_y), tensor_of(b_s1), t_sc);

            if (st != HELIX_OK) {
                suite.check(false, "encode failed: " + tag + " -- " +
                                   helix_ctx_last_error(gpu.ctx));
                continue;
            }

            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                suite.check(false, "GPU error: " + tag + " -- " +
                                   cb.error.localizedDescription.UTF8String);
                continue;
            }

            std::vector<float> y_gpu(p.y_elems()), s_gpu(p.s_elems());
            std::memcpy(y_gpu.data(), b_y.contents,  y_gpu.size() * 4);
            std::memcpy(s_gpu.data(), b_s1.contents, s_gpu.size() * 4);

            suite.check_error(compare(y_gpu, y_ref, rel_tol, abs_tol), "y  " + btag);
            // The carried state never passes through bf16 -- Pass A and B are
            // fp32 on both backends -- so it is held to the fp32 bound even on
            // the MPP path. If that ever starts failing, the bf16 staging has
            // leaked into the carry.
            suite.check_error(compare(s_gpu, s_ref, kRelTolF32, kAbsTolF32), "s1 " + btag);
        }
      }
    }

    helix_ctx_free(gpu.ctx);
    return suite.finish();
}
