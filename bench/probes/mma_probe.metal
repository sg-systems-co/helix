#include <metal_stdlib>
using namespace metal;

// Raw simdgroup MMA throughput probe.
//
// NACC independent accumulator chains run in parallel so the measurement is
// issue-rate bound rather than latency bound. With a single chain every MMA
// waits on the previous one's result and the probe reports latency, which is
// nearly type-independent and gives a badly misleading answer.
#define ITERS 1024
#define NACC  8

template <typename OperandT, typename AccumT>
inline void mma_loop(device AccumT* out, uint gid, threadgroup AccumT* scratch) {
    simdgroup_matrix<AccumT, 8, 8> acc[NACC];
    for (uint k = 0; k < NACC; ++k)
        acc[k] = make_filled_simdgroup_matrix<AccumT, 8, 8>(AccumT(0));

    simdgroup_matrix<OperandT, 8, 8> a =
        make_filled_simdgroup_matrix<OperandT, 8, 8>(OperandT(1.0001f));
    simdgroup_matrix<OperandT, 8, 8> b =
        make_filled_simdgroup_matrix<OperandT, 8, 8>(OperandT(0.9999f));

    for (uint i = 0; i < ITERS; ++i) {
#pragma unroll
        for (uint k = 0; k < NACC; ++k)
            simdgroup_multiply_accumulate(acc[k], a, b, acc[k]);
    }

    // Keep every chain live without touching simdgroup_matrix internals:
    // store each accumulator in turn and fold through threadgroup memory.
    AccumT sink = AccumT(0);
    for (uint k = 0; k < NACC; ++k) {
        simdgroup_store(acc[k], scratch, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        sink += scratch[0];
    }
    if ((gid & 31) == 0) out[gid >> 5] = sink;
}

#define PROBE(NAME, OP, AC)                                                   \
kernel void NAME(device AC* o [[buffer(0)]],                                  \
                 uint g [[thread_position_in_grid]]) {                        \
    threadgroup AC scratch[64];                                               \
    mma_loop<OP, AC>(o, g, scratch);                                          \
}

PROBE(mma_f32_f32,   float,  float)
PROBE(mma_f16_f32,   half,   float)
PROBE(mma_bf16_f32,  bfloat, float)
PROBE(mma_f16_f16,   half,   half)
PROBE(mma_bf16_bf16, bfloat, bfloat)

// Plain vector-FMA probe, for comparison against the MMA numbers above.
// If simdgroup MMA is backed by dedicated matrix hardware it should clear the
// vector ALU by a wide margin. If it lands at roughly the same TFLOP/s, MMA is
// being emulated on the ordinary ALU pipeline -- which would explain why
// reduced precision is SLOWER there (it pays a conversion to fp32 per operand).
template <typename T>
inline void fma_loop(device T* out, uint gid) {
    vec<T,4> a[NACC], b, c;
    for (uint k = 0; k < NACC; ++k) a[k] = vec<T,4>(T(gid + k));
    b = vec<T,4>(T(1.0001f));
    c = vec<T,4>(T(0.9999f));
    for (uint i = 0; i < ITERS * 8; ++i) {
#pragma unroll
        for (uint k = 0; k < NACC; ++k) a[k] = fma(a[k], b, c);
    }
    vec<T,4> s = a[0];
    for (uint k = 1; k < NACC; ++k) s += a[k];
    if (gid == 0xFFFFFFFu) out[0] = s.x + s.y + s.z + s.w;
}

kernel void fma_f32(device float* o [[buffer(0)]], uint g [[thread_position_in_grid]]) { fma_loop<float>(o, g); }
kernel void fma_f16(device half*  o [[buffer(0)]], uint g [[thread_position_in_grid]]) { fma_loop<half>(o, g); }
