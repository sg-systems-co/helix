// test_algebra.cpp -- Milestone 1 gate.
//
// Proves the chunked/reassociated SSD formulation is algebraically identical
// to the sequential recurrence, in fp64, before any MSL exists. If this fails,
// the math is wrong and no amount of kernel debugging will help.
//
// Both sides compute in double and round only on output, so agreement should
// be at f32 round-off (~1e-7 rel), not at some hand-waved kernel tolerance.

#include <cstdio>
#include <vector>

#include "harness.hpp"
#include "helix/reference.hpp"

using namespace helix;
using namespace helix::test;

namespace {

struct Shape {
    int32_t L, n_seq, H, P, N, G, chunk;
    const char* label;
};

// Deliberately includes every sequence-length edge case the plan names:
// L < chunk, L == chunk, L == chunk+1, and L not a multiple of chunk.
const Shape kShapes[] = {
    {   1, 1,  1,  64, 128, 1,  64, "L=1 single token"          },
    {   7, 1,  2,  64, 128, 1,  64, "L=7 partial chunk"         },
    {  63, 1,  2,  64, 128, 1,  64, "L=63 chunk-1"              },
    {  64, 1,  2,  64, 128, 1,  64, "L=64 exact chunk"          },
    {  65, 1,  2,  64, 128, 1,  64, "L=65 chunk+1"              },
    { 127, 1,  2,  64, 128, 1,  64, "L=127 ragged tail"         },
    { 128, 1,  2,  64, 128, 1,  64, "L=128 two chunks"          },
    { 129, 1,  2,  64, 128, 1,  64, "L=129 ragged tail"         },
    { 256, 1,  4,  64,  64, 1,  64, "d_state=64"                },
    { 256, 1,  4, 128, 128, 1,  64, "head_dim=128 (ggml falls back)" },
    { 256, 1,  8, 128, 128, 8,  64, "n_group=8 grouped"         },
    { 256, 4,  4,  64, 128, 1,  64, "n_seq=4 batched"           },
    { 512, 1,  2,  64, 128, 1,  32, "chunk=32"                  },
    { 512, 1,  2,  64, 128, 1, 128, "chunk=128"                 },
    {1024, 1,  2,  64, 128, 1,  64, "L=1024 sixteen chunks"     },
};

// Both references compute in fp64 and round only on the f32 store, so the
// bar is f32 round-off, not a kernel tolerance. Anything looser would let a
// genuinely wrong reassociation through.
constexpr double kRelTol = 1e-6;
constexpr double kAbsTol = 1e-9;

}  // namespace

int main() {
    Suite suite("test_algebra");

    for (const Shape& sh : kShapes) {
        Problem p = Problem::make(sh.L, sh.n_seq, sh.H, sh.P, sh.N, sh.G, sh.chunk, 12345);

        std::vector<float> y_seq(p.y_elems()), s_seq(p.s_elems());
        std::vector<float> y_chk(p.y_elems()), s_chk(p.s_elems());

        ref::scan(p.d, p.s0.data(), p.x.data(), p.dt.data(), p.A.data(),
                  p.B.data(), p.C.data(), p.ids.data(), y_seq.data(), s_seq.data());

        ref::scan_chunked(p.d, p.s0.data(), p.x.data(), p.dt.data(), p.A.data(),
                          p.B.data(), p.C.data(), p.ids.data(), y_chk.data(), s_chk.data());

        // Both sides are fp64 internally; the only loss is the f32 store, so
        // hold them to f32 round-off rather than a loose kernel tolerance.
        suite.check_error(compare(y_chk, y_seq, kRelTol, kAbsTol),
                          std::string("y  ") + sh.label);
        suite.check_error(compare(s_chk, s_seq, kRelTol, kAbsTol),
                          std::string("s1 ") + sh.label);
    }

    // Chunk size must not change the answer: the reassociation is exact, so
    // sweeping the chunk size over a fixed problem is a strong independent
    // check that the inter-chunk state hand-off is right.
    {
        std::vector<float> y_ref, s_ref;
        for (int32_t chunk : {16, 32, 64, 128, 256}) {
            Problem p = Problem::make(509, 1, 4, 64, 128, 1, chunk, 777);
            std::vector<float> y(p.y_elems()), s(p.s_elems());
            ref::scan_chunked(p.d, p.s0.data(), p.x.data(), p.dt.data(), p.A.data(),
                              p.B.data(), p.C.data(), p.ids.data(), y.data(), s.data());
            if (y_ref.empty()) { y_ref = y; s_ref = s; continue; }
            suite.check_error(compare(y, y_ref, kRelTol, kAbsTol),
                              "chunk-invariance y  chunk=" + std::to_string(chunk));
            suite.check_error(compare(s, s_ref, kRelTol, kAbsTol),
                              "chunk-invariance s1 chunk=" + std::to_string(chunk));
        }
    }

    return suite.finish();
}
