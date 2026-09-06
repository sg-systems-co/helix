# Hardware probes

Standalone microbenchmarks that answer "what is this machine actually capable
of", kept out of the main build because they compile with different flags
(`mpp_probe` needs `-std=metal4.0`) and are run by hand, not in CI.

They exist because the project plan carried an assumption -- "bf16 MMA is ~2x
fp32" -- that is false on M5, and acting on it would have made the kernel
slower. Results and interpretation: `docs/PRECISION.md`.

```sh
# simdgroup_matrix and plain vector FMA, per dtype
xcrun -sdk macosx metal -std=metal3.2 -O3 -c mma_probe.metal -o /tmp/mma.air
xcrun -sdk macosx metal /tmp/mma.air -o /tmp/mma.metallib
clang++ -std=c++20 -ObjC++ -O2 mma_probe.mm -framework Metal -framework Foundation -o /tmp/mma
/tmp/mma /tmp/mma.metallib

# MPP tensor ops (M5 Neural Accelerators)
xcrun -sdk macosx metal -std=metal4.0 -O3 -c mpp_probe.metal -o /tmp/mpp.air
xcrun -sdk macosx metal /tmp/mpp.air -o /tmp/mpp.metallib
clang++ -std=c++20 -ObjC++ -O2 mpp_host.mm -framework Metal -framework Foundation -o /tmp/mpp
/tmp/mpp /tmp/mpp.metallib
```

`mma_probe` runs 8 independent accumulator chains on purpose: with a single
chain it measures MMA *latency*, which is nearly dtype-independent and gives a
badly misleading answer.
