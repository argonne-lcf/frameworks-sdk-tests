# PyTorch Distributed collective microbenchmarks

This directory contains the c10d-only collective benchmark payload. The archived
`torchcomms` package and its c10d compatibility adapter are no longer required.

The registered `benchmark-c10d-perf` case launches two local ranks and runs a
bounded all-reduce through native PyTorch Distributed. Each process binds its
XPU from the launcher-provided local rank before the process group is
initialized. Benchmark exceptions are not converted to successful exits.

Run the registered case from the repository root:

```bash
./run_tests run --suite benchmark --id 'benchmark-c10d-perf'
```

For a manual sweep, set `TEST_DEVICE` and `TEST_BACKEND`, then use `torchrun`:

```bash
TEST_DEVICE=xpu TEST_BACKEND=xccl \
  torchrun --standalone --nproc-per-node=2 \
  tests/workloads/collectives/perf/collective_perf_test.py all_reduce \
  --warmup 1 --iters 3 --min-size 4 --max-size 1024
```

`--c10d` remains accepted for compatibility with existing invocations; native
c10d is always used. Pass `--help` for the full collective and option list.
Larger sweeps should be launched with explicit resource limits appropriate to
the target system.
