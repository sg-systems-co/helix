#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// MPP tensor-ops throughput probe.
//
// simdgroup_matrix on M5 measures at roughly plain-vector-FMA rate, which
// suggests it is not touching the per-core Neural Accelerators at all. If that
// is right, MPP matmul2d -- the only documented route to them -- should clear
// it by a wide margin. The accumulator is kept in a cooperative_tensor so the
// loop stays compute bound rather than streaming C through device memory.

#define MM 64
#define NN 64
#define KK 64
#define REPS 1024

template <typename OperandT, typename AccumT>
inline void mpp_loop(device OperandT* Ap, device OperandT* Bp, device AccumT* Cp) {
    constexpr auto d = matmul2d_descriptor(MM, NN, KK, false, false, false);
    matmul2d<d, execution_simdgroups<4>> op;

    using ext2 = dextents<int32_t, 2>;
    tensor<device OperandT, ext2, tensor_inline> tA(Ap, ext2(KK, MM));
    tensor<device OperandT, ext2, tensor_inline> tB(Bp, ext2(NN, KK));
    tensor<device AccumT,   ext2, tensor_inline> tC(Cp, ext2(NN, MM));

    auto cT = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), AccumT>();
#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) cT[i] = AccumT(0);

    for (uint r = 0; r < REPS; ++r)
        op.run(tA, tB, cT);

    cT.store(tC);
}

kernel void mpp_f32(device float*  A [[buffer(0)]], device float*  B [[buffer(1)]],
                    device float*  C [[buffer(2)]]) { mpp_loop<float,  float>(A, B, C); }
kernel void mpp_f16(device half*   A [[buffer(0)]], device half*   B [[buffer(1)]],
                    device float*  C [[buffer(2)]]) { mpp_loop<half,   float>(A, B, C); }
kernel void mpp_bf16(device bfloat* A [[buffer(0)]], device bfloat* B [[buffer(1)]],
                     device float*  C [[buffer(2)]]) { mpp_loop<bfloat, float>(A, B, C); }
