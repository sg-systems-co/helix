// bench_scan.mm -- GPU-timed benchmark for the HELIX scan.
//
// Timing comes from the command buffer's GPUStartTime/GPUEndTime, never from
// wall clock around commit/waitUntilCompleted: the latter measures submission
// and scheduling latency, which at these kernel durations dominates the signal.
//
// Two bandwidth figures are reported deliberately:
//
//   algorithmic  the bytes the algorithm must move (inputs + outputs, once)
//   actual       that plus the per-chunk state spill to device memory
//
// The gap between them is exactly what the Week 3 on-chip-state work removes,
// so it is worth watching as a number rather than inferring it from a speedup.

#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstdint>
#include <vector>
#include <cmath>
#include <chrono>

#include "tests/harness.hpp"
#include "helix/helix.h"
#include "src/kernels/helix_params.h"

#define HELIX_CS_BENCH HELIX_CS

using namespace helix::test;

namespace {

struct Config {
    int32_t L = 4096, n_seq = 1, H = 64, P = 64, N = 128, G = 8;
    int     iters = 50;
    std::string impls = "1pass,3pass";
    // Apple GPUs ramp their clock with sustained load. A fixed iteration count
    // measures whatever power state the process happened to start in: the same
    // config was observed at 11.6 ms and 34.2 ms across runs, a 3x spread that
    // would make any speedup claim meaningless. Warm up on wall-clock time
    // instead, until the GPU has been busy long enough to have ramped.
    double  warmup_ms = 800.0;
    bool    json = false;
};

struct Dist {
    double min = 0, median = 0, max = 0;
};

Dist distribution(std::vector<double> v) {
    Dist d;
    if (v.empty()) return d;
    std::sort(v.begin(), v.end());
    d.min    = v.front();
    d.median = v[v.size() / 2];
    d.max    = v.back();
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const size_t eq = a.find('=');
        const int32_t v = (eq == std::string::npos) ? 0 : (int32_t)std::atoi(a.c_str() + eq + 1);
        if      (a.rfind("--L=", 0) == 0)        cfg.L     = v;
        else if (a.rfind("--n-seq=", 0) == 0)    cfg.n_seq = v;
        else if (a.rfind("--heads=", 0) == 0)    cfg.H     = v;
        else if (a.rfind("--head-dim=", 0) == 0) cfg.P     = v;
        else if (a.rfind("--d-state=", 0) == 0)  cfg.N     = v;
        else if (a.rfind("--groups=", 0) == 0)   cfg.G     = v;
        else if (a.rfind("--iters=", 0) == 0)    cfg.iters = v;
        else if (a.rfind("--warmup-ms=", 0) == 0) cfg.warmup_ms = v;
        else if (a.rfind("--impl=", 0) == 0)     cfg.impls = a.substr(7);
        else if (a == "--json")                  cfg.json  = true;
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { std::fprintf(stderr, "no Metal device\n"); return 1; }
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    helix_ctx* ctx = nullptr;
    if (helix_ctx_create(&ctx, (__bridge void*)dev, nullptr) != HELIX_OK) {
        std::fprintf(stderr, "helix_ctx_create failed\n");
        return 1;
    }

    Problem prob = Problem::make(cfg.L, cfg.n_seq, cfg.H, cfg.P, cfg.N, cfg.G, 0, 99);

    auto up = [&](const void* src, size_t bytes) {
        return [dev newBufferWithBytes:src length:bytes options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> b_s0 = up(prob.s0.data(), prob.s0.size() * 4);
    id<MTLBuffer> b_x  = up(prob.x.data(),  prob.x.size()  * 4);
    id<MTLBuffer> b_dt = up(prob.dt.data(), prob.dt.size() * 4);
    id<MTLBuffer> b_A  = up(prob.A.data(),  prob.A.size()  * 4);
    id<MTLBuffer> b_B  = up(prob.B.data(),  prob.B.size()  * 4);
    id<MTLBuffer> b_C  = up(prob.C.data(),  prob.C.size()  * 4);
    id<MTLBuffer> b_y  = [dev newBufferWithLength:prob.y_elems() * 4
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_s1 = [dev newBufferWithLength:prob.s_elems() * 4
                                          options:MTLResourceStorageModeShared];

    auto T = [](id<MTLBuffer> b) { return helix_tensor{(__bridge void*)b, 0}; };

    // One variant per implementation under test. They are measured interleaved
    // in this single process, never in separate runs: the GPU power state is
    // global and sticky, so cross-process A/B is not salvageable by averaging.
    // See docs/BENCHMARKING.md.
    struct Variant {
        std::string          name;
        helix_scan_desc      desc{};
        id<MTLBuffer>        scratch = nil;
        std::vector<double>  times;
        bool                 ok = true;
    };

    std::vector<Variant> variants;
    {
        std::string list = cfg.impls;
        size_t pos = 0;
        while (pos <= list.size()) {
            const size_t comma = list.find(',', pos);
            const std::string name = list.substr(pos, comma == std::string::npos
                                                      ? std::string::npos : comma - pos);
            if (!name.empty()) {
                Variant v;
                v.name = name;
                v.desc = prob.d;
                v.desc.multipass = (name == "3pass");
                if (!helix_scan_supported(ctx, &v.desc)) {
                    std::fprintf(stderr, "shape unsupported for impl '%s'\n", name.c_str());
                    return 1;
                }
                const size_t sc = helix_scan_scratch_size(ctx, &v.desc);
                if (sc) v.scratch = [dev newBufferWithLength:sc
                                                    options:MTLResourceStorageModePrivate];
                variants.push_back(v);
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
    if (variants.empty()) { std::fprintf(stderr, "no impls selected\n"); return 1; }

    auto run_once = [&](Variant& v, double* gpu_ms) -> bool {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            const helix_tensor sc = v.scratch ? T(v.scratch) : HELIX_TENSOR_NULL;
            if (helix_scan_encode(ctx, (__bridge void*)cb, &v.desc,
                                  T(b_s0), T(b_x), T(b_dt), T(b_A), T(b_B), T(b_C),
                                  HELIX_TENSOR_NULL, T(b_y), T(b_s1), sc) != HELIX_OK) {
                std::fprintf(stderr, "encode failed (%s): %s\n",
                             v.name.c_str(), helix_ctx_last_error(ctx));
                return false;
            }
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                std::fprintf(stderr, "GPU error (%s): %s\n", v.name.c_str(),
                             cb.error.localizedDescription.UTF8String);
                return false;
            }
            if (gpu_ms) *gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
        }
        return true;
    };

    // Warm up on wall-clock time, cycling every variant, so the clock has
    // settled and no variant is measured from a colder state than another.
    {
        const auto t0 = std::chrono::steady_clock::now();
        int rounds = 0;
        for (;;) {
            for (auto& v : variants) if (!run_once(v, nullptr)) return 1;
            ++rounds;
            const double el = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            if (el >= cfg.warmup_ms && rounds >= 3) break;
        }
    }

    for (int it = 0; it < cfg.iters; ++it)
        for (auto& v : variants) {
            double t = 0.0;
            if (!run_once(v, &t)) return 1;
            v.times.push_back(t);
        }

    // Derived quantities are properties of the problem, not the kernel.
    const double L = cfg.L, H = cfg.H, P = cfg.P, N = cfg.N, G = cfg.G, S = cfg.n_seq;
    const double CS = HELIX_CS_BENCH;
    const double n_chunks = std::floor((L + CS - 1) / CS);
    const double tri   = 0.5 * CS * (CS + 1);
    const double flops = 2.0 * S * H * n_chunks * (tri * N + tri * P + 2.0 * CS * N * P);

    if (!cfg.json) {
        std::printf("L=%d H=%d P=%d N=%d G=%d seq=%d   threadgroups: 1pass=%.0f 3pass<=%.0f\n",
                    cfg.L, cfg.H, cfg.P, cfg.N, cfg.G, cfg.n_seq,
                    std::ceil(P / (double)HELIX_HD) * H * S,
                    std::ceil(P / (double)HELIX_HD) * H * S * n_chunks);
        std::printf("  %-8s %10s %12s %11s %9s %9s\n",
                    "impl", "ms", "tok/s", "GFLOP/s", "spread", "vs 1pass");
    }

    double base_ms = 0.0;
    for (auto& v : variants) {
        const Dist d = distribution(v.times);
        const double sec = d.min * 1e-3;
        if (v.name == "1pass") base_ms = d.min;
        const double rel = base_ms > 0 ? base_ms / d.min : 0.0;

        if (cfg.json) {
            std::printf("{\"impl\":\"%s\",\"L\":%d,\"n_seq\":%d,\"heads\":%d,"
                        "\"head_dim\":%d,\"d_state\":%d,\"n_group\":%d,\"ms\":%.4f,"
                        "\"ms_median\":%.4f,\"spread\":%.3f,\"tok_per_s\":%.1f,"
                        "\"gflops\":%.1f,\"speedup_vs_1pass\":%.3f}\n",
                        v.name.c_str(), cfg.L, cfg.n_seq, cfg.H, cfg.P, cfg.N, cfg.G,
                        d.min, d.median, d.max / d.min, L * S / sec,
                        flops / sec * 1e-9, rel);
        } else {
            std::printf("  %-8s %10.3f %12.0f %11.0f %8.2fx %8s\n",
                        v.name.c_str(), d.min, L * S / sec, flops / sec * 1e-9,
                        d.max / d.min,
                        rel > 0 ? (std::to_string(rel).substr(0, 5) + "x").c_str() : "-");
        }
    }

    helix_ctx_free(ctx);
    return 0;
}
