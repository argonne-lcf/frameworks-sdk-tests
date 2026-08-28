# Frameworks SDK validation tests

This repository consolidates the tests used to validate the ALCF Frameworks
SDK after `module load frameworks`. Test source and small required fixtures are
kept here; virtual environments, upstream source checkouts, compiler caches,
profiles, logs, checkpoints, core dumps, and large datasets are not.

The default run is deliberately small. It loads the requested module, checks
the core Python packages, exercises PyTorch/XCCL on one XPU, validates MPI,
and runs dpctl/dpnp checks. Larger distributed tests, regressions, application
workloads, and performance benchmarks must be selected explicitly.

## Quick start

List everything in the catalog:

```bash
./run_tests list
```

Check prerequisites for the default smoke suite without running it:

```bash
./run_tests doctor --module frameworks
```

Run the default smoke suite:

```bash
./run_tests run --module frameworks
```

Use a versioned or staging module by passing its exact Lmod name:

```bash
./run_tests run --module frameworks/2026.1.0
```

The runner uses a login Bash process to load the module and captures the
resulting environment. To test an environment that is already active:

```bash
./run_tests run --no-module
```

On a host without an XPU, hardware tests are reported as `SKIP`; package and
MPI checks still run. XPU capacity is detected with the loaded PyTorch build,
with independent dpctl/Level Zero discovery as a fallback so a broken PyTorch
XPU integration cannot hide available hardware.
`--available-xpus` and `--available-nodes` exist for environments where
automatic resource detection is unavailable, not for bypassing real resource
requirements.

## Selecting tests

Suites are intentionally separated by cost and purpose:

- `smoke`: default package and single-XPU acceptance checks.
- `optional-imports`: science, LLM, communication, and Intel/IPEX package groups.
- `harness`: CPU/gloo and fault-injection checks for the collective validator.
- `distributed`: two- and four-XPU PyTorch/XCCL collectives, including
  subgroup communicator layouts, plus direct GPU-aware MPI.
- `regression`: SDPA, JIT, vLLM, Gamma sampling, and XCCL memory regressions.
- `workload`: bounded DDP/FSDP/1-D and 2-D DTensor, DeepSpeed, TorchComms,
  sequence parallelism, MoE, and split CosmicTagger application tests.
- `benchmark`: opt-in bounded TorchComms/c10d collectives and GEMM performance
  experiments.

Examples:

```bash
./run_tests run --suite distributed --module frameworks
./run_tests run --suite regression --tag 'sdpa' --module frameworks
./run_tests run --id 'workload-*ddp' --module frameworks
./run_tests run --all --dry-run --module frameworks
```

Repeated selectors are ORed within that selector type; suite, tag, and ID
selectors are ANDed with each other. `--all` cannot be combined with a
selector. Be careful with a real `--all` run: it includes multi-XPU tests and
long benchmarks.

Every run gets a unique directory below `results/`, containing one combined
stdout/stderr log and a per-case `artifacts/` directory for each test, plus a
machine-readable `summary.json`. The summary records the requested module and
the resolved `LOADEDMODULES` list. A failing or timed-out test makes the runner
exit nonzero; so does a real run in which every selected test skips. Missing
declared prerequisites produce an explicit skip rather than a false pass.

## PBS multi-node collectives

The PyTorch cases in the direct `distributed` suite use `torchrun`; its
GPU-aware MPI case uses `mpiexec`. For a real multi-node PALS/PBS fabric test,
use the consolidated PyTorch launcher:

```bash
TEST_CASE=allreduce \
FRAMEWORKS_MODULE=frameworks/2026.1.0 \
PROJECT=datascience QUEUE=workq NNODES=2 \
./scripts/run_torch_collective_pbs.sh --submit
```

`TEST_CASE` may be `allreduce`, `allgather`, `alltoall`,
`alltoall_uneven`, `reduce_scatter`, `overlap`, `p2p`, `streams`, or
`subgroups`. For `subgroups`, `TEST_GROUPS` may be `ep`, `pp`, `seq`,
`disjoint`, or `overlap`.
`NRANKS_PER_NODE`, `XPUS_PER_NODE`, `WALLTIME`, `TEST_MEM_BUDGET_GB`,
`TEST_ITERS`, and the operation-specific `TEST_*` variables are overrideable.
The script contains no personal or `/lus/.../testing` paths.

## Layout

```text
runner.py / run_tests       dependency-free harness
suite.json                  test catalog, requirements, timeouts, and tags
harness_tests/              unit tests for the harness itself
scripts/                    shared PBS and Aurora affinity helpers
tests/smoke/                fast module acceptance checks
tests/distributed/          scalable correctness-aware collective tests
tests/regressions/          focused known-bug reproducers
tests/workloads/            model/application tests and opt-in benchmarks
docs/                       migration record and manual-test catalog
```

See [docs/test-catalog.md](docs/test-catalog.md) for tests that need manual
arguments or specialized launchers, and
[docs/source-inventory.md](docs/source-inventory.md) for provenance and the
explicit exclusion list.

## Adding a test

Add self-contained source under the appropriate `tests/` category, then add a
case to `suite.json`. Commands must be JSON argument arrays; they are never
interpreted by a shell. Declare resource and dependency prerequisites instead
of detecting them by exiting successfully. Tests must return nonzero when
their validation fails and should write generated output under a temporary or
results directory.
