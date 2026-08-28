# TorchComms collective microbenchmarks

This directory contains the portable installed-package payload from
`torchcomm_example/torchcomms-perf-alcf/perf_standalone`. Site setup, PBS
submission, result-analysis, upstream-source, log, and generated-result files
are intentionally excluded.

The registered `benchmark` cases launch two local ranks and run a bounded
all-reduce through each adapter:

- native `torchcomms`
- PyTorch `c10d`
- PyTorch `c10d` routed through TorchComms

Each process maps torchrun's rank variables to the names consumed by
TorchComms/XCCL, then binds its XPU from the launcher-provided local rank before
the communicator is initialized. Benchmark exceptions are not converted to
successful exits.

Run the registered cases from the repository root:

```bash
./run_tests run --suite benchmark --id '*torchcomms-perf*'
./run_tests run --suite benchmark --id 'benchmark-c10d-perf'
```

For a manual sweep, set `TEST_DEVICE` and `TEST_BACKEND`, then use `torchrun`:

```bash
TEST_DEVICE=xpu TEST_BACKEND=xccl \
  torchrun --standalone --nproc-per-node=2 \
  tests/workloads/torchcomms/perf/collective_perf_test.py all_reduce \
  --warmup 1 --iters 3 --min-size 4 --max-size 1024
```

Use `--c10d` or `--c10d-torchcomms` to select another adapter. Pass `--help`
for the full collective and option list. Larger sweeps should be launched with
explicit resource limits appropriate to the target system.
