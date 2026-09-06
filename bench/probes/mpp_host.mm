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
    if (!lib) { printf("lib load: %s\n", e.localizedDescription.UTF8String); return 1; }

    const NSUInteger TG = 128, NTG = 4096;   // 4 simdgroups per TG, as the descriptor requires
    const double REPS = 1024;
    // Each op.run() is a 64x64x64 matmul per threadgroup = 2*64^3 flops.
    const double flops = (double)NTG * REPS * 2.0 * 64.0 * 64.0 * 64.0;

    id<MTLBuffer> A = [dev newBufferWithLength:64*64*4 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> B = [dev newBufferWithLength:64*64*4 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> C = [dev newBufferWithLength:64*64*4 options:MTLResourceStorageModePrivate];

    printf("  %-12s %10s %12s\n", "variant", "ms", "TFLOP/s");
    for (const char* n : {"mpp_f32","mpp_f16","mpp_bf16"}) {
      id<MTLFunction> fn = [lib newFunctionWithName:@(n)];
      if (!fn) { printf("  %-12s missing\n", n); continue; }
      id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&e];
      if (!pso) { printf("  %-12s pipeline: %s\n", n, e.localizedDescription.UTF8String); continue; }
      auto once = [&](double* ms){
        @autoreleasepool {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:pso];
          [enc setBuffer:A offset:0 atIndex:0];
          [enc setBuffer:B offset:0 atIndex:1];
          [enc setBuffer:C offset:0 atIndex:2];
          [enc dispatchThreadgroups:MTLSizeMake(NTG,1,1) threadsPerThreadgroup:MTLSizeMake(TG,1,1)];
          [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
          if (cb.error) { printf("  gpu err: %s\n", cb.error.localizedDescription.UTF8String); }
          if (ms) *ms = (cb.GPUEndTime - cb.GPUStartTime)*1e3;
        }
      };
      auto t0 = std::chrono::steady_clock::now();
      for(;;){ once(nullptr);
        if (std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count() > 1200) break; }
      std::vector<double> ts;
      for (int i=0;i<25;i++){ double m; once(&m); ts.push_back(m); }
      std::sort(ts.begin(), ts.end());
      printf("  %-12s %10.3f %12.2f\n", n, ts.front(), flops/(ts.front()*1e-3)*1e-12);
    }
  }
  return 0;
}
