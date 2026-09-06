// test_drift.mm -- numerical drift across many chunk boundaries.
//
// Per-chunk error being small says nothing about whether it compounds. This
// walks 8192 tokens (128 chunks) and checks the carried state against the fp64
// reference at boundaries all the way along, so a scheme that is fine for one
// chunk but accumulates across a hundred fails here rather than in a perplexity
// run three milestones later.
//
// It is the specific guard on two decisions: moving G4 onto the matrix unit
// (upstream ggml-metal keeps that reduction in token order precisely because
// reassociating it "compounds rounding differences at every chunk boundary"),
// and any future move of the carried state to reduced precision.
//
// Boundary states are read through the public API by re-running the scan
// truncated to each boundary -- HELIX exposes no intermediate state, and a test
// that reached inside for one would not be testing the shipped path.

#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "harness.hpp"
#include "helix/helix.h"
#include "helix/reference.hpp"

using namespace helix;
using namespace helix::test;

namespace {

// Same calibrated bounds as the parity test. The point here is not a looser
// tolerance for long sequences -- it is that the tolerance must still hold at
// chunk 127 having held at chunk 0.
constexpr double kRelTol = 5e-6;
constexpr double kAbsTol = 5e-7;

// The carried state is fp32 on both backends -- Pass A and Pass B never touch
// bf16, and the MPP path narrows only Pass C's GEMM operands. So the state is
// held to the fp32 bound even when MPP is running, and this test doubles as the
// check that bf16 staging has not leaked into the carry.

constexpr int32_t kTokens  = 8192;
constexpr int32_t kChunk   = 64;
constexpr int32_t kHeads   = 8;
constexpr int32_t kHeadDim = 64;
constexpr int32_t kState   = 128;

struct Mode { const char* label; bool multipass; Memory mem; };

// Both memory regimes. SHORT stresses the intra-chunk math; LONG is the only
// one that actually exercises the inter-chunk state carry, because under SHORT
// the per-chunk decay is ~1e-14 and the carry is numerically irrelevant.
const Mode kModes[] = {
    {"1pass/short", false, Memory::SHORT},
    {"3pass/short", true,  Memory::SHORT},
    {"1pass/long",  false, Memory::LONG },
    {"3pass/long",  true,  Memory::LONG },
};

}  // namespace

int main() {
    Suite suite("test_drift");

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { std::printf("SKIP: no Metal device\n"); return 0; }
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    helix_ctx* ctx = nullptr;
    if (helix_ctx_create(&ctx, (__bridge void*)dev, nullptr) != HELIX_OK) {
        std::printf("SKIP: helix_ctx_create failed\n");
        return 0;
    }

    const int32_t n_chunks = (kTokens + kChunk - 1) / kChunk;
    std::printf("L=%d  chunks=%d  H=%d P=%d N=%d\n", kTokens, n_chunks,
                kHeads, kHeadDim, kState);

    for (const Mode& mode : kModes) {
      Problem p = Problem::make(kTokens, 1, kHeads, kHeadDim, kState, 1, kChunk,
                                20260905, mode.mem);

      // fp64 ground truth plus the state at every chunk boundary, one pass.
      std::vector<float> y_ref(p.y_elems()), s_ref(p.s_elems());
      std::vector<std::vector<double>> snaps;
      ref::scan_with_snapshots(p.d, p.s0.data(), p.x.data(), p.dt.data(), p.A.data(),
                               p.B.data(), p.C.data(), p.ids.data(),
                               y_ref.data(), s_ref.data(), kChunk, &snaps);

      @autoreleasepool {
        id<MTLBuffer> b_s0 = [dev newBufferWithBytes:p.s0.data() length:p.s0.size()*4
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_x  = [dev newBufferWithBytes:p.x.data()  length:p.x.size()*4
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_dt = [dev newBufferWithBytes:p.dt.data() length:p.dt.size()*4
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_A  = [dev newBufferWithBytes:p.A.data()  length:p.A.size()*4
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_B  = [dev newBufferWithBytes:p.B.data()  length:p.B.size()*4
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_C  = [dev newBufferWithBytes:p.C.data()  length:p.C.size()*4
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_y  = [dev newBufferWithLength:p.y_elems()*4
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_s1 = [dev newBufferWithLength:p.s_elems()*4
                                              options:MTLResourceStorageModeShared];
        auto T = [](id<MTLBuffer> b) { return helix_tensor{(__bridge void*)b, 0}; };

        {
            std::printf("  %s:\n", mode.label);

            std::vector<double> drift;      // max-abs state error at each probe
            std::vector<int32_t> probes;
            bool failed = false;

            // Every 8th boundary plus the last: dense enough to expose growth,
            // cheap enough to stay a unit test.
            for (int32_t c = 7; c < n_chunks; c += 8) {
                probes.push_back(c);

                helix_scan_desc d = p.d;
                d.n_tok     = (c + 1) * kChunk;
                d.multipass = mode.multipass;
                if (!helix_scan_supported(ctx, &d)) {
                    suite.check(false, std::string(mode.label) + " unsupported at chunk " +
                                       std::to_string(c));
                    failed = true;
                    break;
                }

                const size_t sbytes = helix_scan_scratch_size(ctx, &d);
                id<MTLBuffer> b_sc = sbytes
                    ? [dev newBufferWithLength:sbytes options:MTLResourceStorageModePrivate]
                    : nil;

                id<MTLCommandBuffer> cb = [queue commandBuffer];
                const helix_status st = helix_scan_encode(
                    ctx, (__bridge void*)cb, &d,
                    T(b_s0), T(b_x), T(b_dt), T(b_A), T(b_B), T(b_C),
                    HELIX_TENSOR_NULL, T(b_y), T(b_s1),
                    b_sc ? T(b_sc) : HELIX_TENSOR_NULL);
                if (st != HELIX_OK) {
                    suite.check(false, std::string(mode.label) + " encode failed at chunk " +
                                       std::to_string(c));
                    failed = true;
                    break;
                }
                [cb commit];
                [cb waitUntilCompleted];

                const std::vector<double>& want = snaps[c];
                double max_abs = 0.0, worst_ratio = 0.0;
                const float* got = (const float*)b_s1.contents;
                for (size_t i = 0; i < p.s_elems(); ++i) {
                    const double a = got[i], b = want[i];
                    if (!std::isfinite(a)) { worst_ratio = INFINITY; break; }
                    const double err = std::fabs(a - b);
                    max_abs = std::max(max_abs, err);
                    worst_ratio = std::max(worst_ratio,
                                           err / (kAbsTol + kRelTol * std::fabs(b)));
                }
                drift.push_back(max_abs);

                if (worst_ratio > 1.0) {
                    suite.check(false, std::string(mode.label) + " tolerance exceeded at chunk " +
                                       std::to_string(c) + " (ratio " +
                                       std::to_string(worst_ratio) + ")");
                    failed = true;
                }
            }

            if (failed) goto done;

            for (size_t i = 0; i < probes.size(); ++i)
                std::printf("      chunk %3d (tok %5d)  max|ds| = %.3e\n",
                            probes[i], (probes[i] + 1) * kChunk, drift[i]);

            suite.check(true, std::string(mode.label) +
                              " all boundaries within calibrated tolerance");

            // Bounded, not merely small. fp32 round-off accumulates like a
            // random walk, so drift may grow slowly -- but a scheme that
            // compounds error multiplicatively across chunks would show the
            // tail exploding relative to the first quarter. Compare the worst
            // late-sequence drift to the worst early-sequence drift.
            const size_t q = probes.size() / 4;
            double early = 0.0, late = 0.0;
            for (size_t i = 0; i < q; ++i)                     early = std::max(early, drift[i]);
            for (size_t i = probes.size() - q; i < probes.size(); ++i)
                                                               late  = std::max(late,  drift[i]);
            const double growth = early > 0 ? late / early : 1.0;
            std::printf("      early-quarter max %.3e -> late-quarter max %.3e  (%.2fx)\n",
                        early, late, growth);
            suite.check(growth < 10.0,
                        std::string(mode.label) + " drift bounded across 128 chunks (" +
                        std::to_string(growth).substr(0, 4) + "x)");
        }
        done:;
      }
    }

    helix_ctx_free(ctx);
    return suite.finish();
}
