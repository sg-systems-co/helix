#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
int main(int argc, char** argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError* e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e];
    if (!lib) { printf("lib load failed: %s\n", e.localizedDescription.UTF8String); return 1; }

    const NSUInteger TG = 256, NTG = 2048;      // 2048 threadgroups x 8 simdgroups
    const NSUInteger simdgroups = NTG * (TG/32);
    const double ITERS = 1024, MMA_PER_ITER = 8;
    // Each 8x8x8 MMA is 8*8*8 multiply-adds = 1024 flops.
    const double flops = (double)simdgroups * ITERS * MMA_PER_ITER * 8*8*8 * 2;

    id<MTLBuffer> out = [dev newBufferWithLength:simdgroups*4 options:MTLResourceStorageModePrivate];
    const char* names[] = {"mma_f32_f32","mma_f16_f32","mma_bf16_f32","mma_f16_f16","mma_bf16_bf16","fma_f32","fma_f16"};

    printf("  %-16s %10s %12s %9s\n", "variant", "ms", "TFLOP/s", "vs f32");
    double base = 0;
    for (const char* n : names) {
      id<MTLFunction> fn = [lib newFunctionWithName:@(n)];
      id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&e];
      if (!pso) { printf("  %-16s pipeline failed\n", n); continue; }
      std::vector<double> ts;
      auto once = [&](double* ms){
        @autoreleasepool {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:pso];
          [enc setBuffer:out offset:0 atIndex:0];
          [enc dispatchThreadgroups:MTLSizeMake(NTG,1,1) threadsPerThreadgroup:MTLSizeMake(TG,1,1)];
          [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
          if (ms) *ms = (cb.GPUEndTime - cb.GPUStartTime)*1e3;
        }
      };
      auto t0 = std::chrono::steady_clock::now();
      for(;;){ once(nullptr);
        if (std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count() > 1200) break; }
      for (int i=0;i<25;i++){ double m; once(&m); ts.push_back(m); }
      std::sort(ts.begin(), ts.end());
      const double ms = ts.front();
      // The FMA probe does ITERS*8 iterations x NACC lanes x 4 components,
      // 2 flops each -- not the 8x8x8 MMA shape.
      const bool is_fma = (n[0]=='f' && n[1]=='m');
      const double f = is_fma
          ? (double)simdgroups*32.0*ITERS*8.0*8.0*4.0*2.0
          : flops;
      const double tf = f/(ms*1e-3)*1e-12;
      if (base==0) base = tf;
      printf("  %-16s %10.3f %12.2f %8.2fx\n", n, ms, tf, tf/base);
    }
  }
  return 0;
}
