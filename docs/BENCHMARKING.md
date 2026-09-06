# Benchmarking HELIX

## The hazard

The same kernel, same shape, same binary, measured minutes apart:

```
run 1: 11.53 ms   514 GFLOP/s
run 2: 12.21 ms   486 GFLOP/s
run 3: 27.19 ms   218 GFLOP/s
run 4: 35.05 ms   169 GFLOP/s
```

A 3x spread with nothing changed. Any "speedup" measured across two such runs
would be pure noise, and it would be noise pointing whichever way you hoped.

The cause is the Apple GPU power state, which is **global and sticky across
processes** and decays with idle time. It is not thermal throttling -- running a
large sustained job immediately *before* the measurement makes it faster, not
slower:

```
after ./bench_scan --L=8192 --n-seq=8:
run 1: 11.55 ms   514 GFLOP/s
run 2: 11.51 ms   515 GFLOP/s
```

## The other hazard: a busy machine

Separately from power state, an unrelated job on the machine will quietly ruin a
run. A background `python` process at 601% CPU took the single-pass kernel from
its true 510 GFLOP/s down to 190, and inflated `spread` to 3.4x. Worse, it did
not hurt both kernels equally: the low-occupancy single-pass kernel is
latency-bound and suffers disproportionately when descheduled, while the
high-occupancy 3-pass kernel saturates whatever GPU time it gets. Contention
therefore *inflates* the measured speedup -- in the flattering direction.

Before any run that produces a number worth quoting:

```sh
ps -Ao %cpu,comm -r | head -4
ioreg -r -d 1 -w 0 -c IOAccelerator | grep -oE '"Device Utilization %"=[0-9]+'
```

`spread` is the built-in tripwire: on a quiet machine it is 1.03-1.07x. Anything
above ~1.15x means discard the run and look at what else is running.

## The protocol

1. **Warm up on wall-clock time, not iteration count.** `bench_scan` runs the
   kernel until `--warmup-ms` (default 800) has elapsed before it records
   anything. A fixed iteration count measures whatever power state the process
   started in.
2. **Report the minimum, print the spread.** The minimum is the least
   clock-contaminated estimate of what the kernel costs. `spread` (max/min) is
   printed beside it: **if spread > ~1.15x, discard the run.** A warmed GPU
   gives 1.02-1.06x consistently.
3. **Never compare across processes.** For any A/B -- HELIX vs upstream, Week 2
   vs Week 3 -- run both in the same process, alternating, so they see the same
   power state. Cross-process comparison is not salvageable by averaging.
4. **Never time with wall clock around `commit`/`waitUntilCompleted`.** Use
   `MTLCommandBuffer.GPUStartTime` / `GPUEndTime`. At these durations, submission
   and scheduling latency dominate wall-clock measurement.
5. **Check the machine is quiet first** (see above), and confirm the run's
   `spread` before believing its numbers.

`bench_scan --impl=1pass,3pass` implements 1-3 directly: it constructs both
variants up front, warms them together, then alternates them iteration by
iteration.

Verification that the protocol works -- nine consecutive runs, three warmup
settings, cold-to-warm:

```
warmup 800ms:  11.53 / 11.52 / 11.51 ms
warmup 3000ms: 11.52 / 11.50 / 11.48 ms
warmup 6000ms: 11.50 / 11.54 / 11.52 ms
```

## Reading the bandwidth figures

`bench_scan` reports two:

- **algorithmic** -- the bytes the algorithm must move: every input read once,
  every output written once.
- **actual** -- that plus the per-chunk state spill to device memory.

The `spill_ratio` between them is the cost of the state not fitting threadgroup
memory in fp32 (a 128x64 fp32 state is 32 KiB, over the whole budget). At
L=8192 it is 2.57x. Driving it to 1.0x is exactly what the Week 3 bf16 on-chip
state does, so it is worth watching directly rather than inferring it from a
wall-clock speedup.
