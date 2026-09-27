# Test catalog notes

`suite.json` is the executable catalog. This document covers retained source
that needs an argument, model, scheduler, or investigation-specific decision
and therefore is not launched by a normal suite selection.

## Registered suites

- `smoke` checks the core package ABI/import surface, PyTorch XPU/XCCL,
  mpi4py, dpctl, dpnp, and PyTorch/dpnp DLPack sharing.
- `optional-imports` checks science, LLM, communication, ezpz, and legacy
  Intel/IPEX package groups without making them part of the default acceptance
  gate.
- `harness` exercises the collective validator on CPU/gloo, through PALS-style
  environment variables, and with expected fault injection.
- `distributed` contains correctness-aware all-reduce, all-gather,
  all-to-all, uneven all-to-all, reduce-scatter, collective/compute overlap,
  five P2P modes, independent-stream overlap, five expert/pipeline/disjoint/
  overlapping subgroup communicator modes, direct GPU-buffer
  `mpi4py.Allreduce` validation, and ezpz distributed bring-up under both
  `mpiexec` and `torchrun`.
- `regression` contains the GQA SDPA compiler crash, its baseline, the full
  TP/FSDP SDPA reproducer, Gamma sampling, DeepSpeed and IPEX JIT builds, vLLM
  registry inspection, and the XCCL `empty_cache` memory leak.
- `workload` contains checkpoint I/O, 1-D and 2-D DTensor redistribution,
  bounded MNIST/ResNet/Transformer training, DeepSpeed miniGPT, TorchComms,
  XPU sequence parallelism, separate MoE reference/Triton pytest gates, a
  bounded MoE training probe, and dependency-split CosmicTagger pytest cases.
- `benchmark` contains bounded two-rank all-reduce measurements for the
  TorchComms, c10d, and c10d-through-TorchComms adapters, GEMM sweeps, and
  bounded `ezpz benchmark` training runs across the s/m/l model ladder.

### ezpz benchmark

`ezpz benchmark` runs ezpz's own example suite and writes `timings.csv`,
`env.json`, and `report.md` per run, exiting nonzero if any example fails —
so the harness gets pass/fail and a machine-readable measurement for free.
The registered cases go through `tests/workloads/ezpz/run_benchmark.sh`, which
exists for two reasons the manifest cannot express.

**Artifact routing.** Test commands are argv arrays and are never templated,
so the per-case artifact path can only be read from
`FRAMEWORKS_TEST_ARTIFACT_DIR` at runtime. Without the wrapper, `ezpz
benchmark` writes into `./outputs/` in the working tree.

**Refusing the one unbounded example.** `ezpz benchmark` forwards only
`--model`, and the examples do not share a bounding flag, so each one was
measured at the defaults `ezpz benchmark` actually uses (one Aurora node, 12
ranks, `--model s`, 900s cap):

| example | bound | measured | registered |
|---|---|---|---|
| `test` | `--train-iters` (400 at s/m/l) | 48s | yes |
| `vit` | `--max-iters` (default 224) | 46s | yes |
| `fsdp` | `--epochs` 10 over MNIST | 53s | yes |
| `diffusion` | `--train-steps` (default 400) | 69s | yes |
| `hf` | `--max-steps` (HF `TrainingArguments`) | 72s | yes, direct launch |
| `hf_trainer` | `--max-steps` (HF `TrainingArguments`) | 42s | yes, direct launch |
| `fsdp_tp` | `--epochs` 5 over full imdb | **TIMEOUT at 900s** | **no** |

Only `fsdp_tp` is genuinely unbounded: its cost scales with a 25k-row dataset
at `seq_len` 2048 and nothing `ezpz benchmark` forwards caps it. `--epochs` by
itself is not the tell — `fsdp` uses it too and finishes in under a minute
because MNIST is a fixed size, and `diffusion` defaults to a toy corpus with
`--train-steps 400`. The wrapper therefore refuses `fsdp_tp` with exit 2 and
points at running it in a dedicated job with an explicit budget:

```bash
ezpz launch -- python3 -m ezpz.examples.fsdp_tp --model s --epochs 1
```

It also treats "exit 0 but no `timings.csv`" as a failure, so an example that
silently produces no measurement cannot pass.

`hf` and `hf_trainer` are registered, but they do not go through `ezpz
benchmark`: that path hard-codes `meta-llama/Llama-3.2-1B`, which is gated (an
anonymous fetch of its config returns 401, versus 307 for Qwen), and
`--report-to=wandb`. Neither is overridable through `ezpz benchmark`, so the
wrapper launches those two modules directly against a public checkpoint with
an explicit step budget, then writes the same `timings.csv` contract itself.
Defaults, all overridable by environment variable:

| variable | default |
|---|---|
| `EZPZ_BENCH_HF_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `EZPZ_BENCH_HF_DATASET` | `eliplutchok/fineweb-small-sample` |
| `EZPZ_BENCH_HF_MAX_STEPS` | `20` |
| `EZPZ_BENCH_HF_BLOCK_SIZE` | `1024` |

The wrapper uses ezpz's documented offline recipe,
`EZPZ_TRACKER_BACKENDS=none WANDB_DISABLED=1` ([configuration →
experiment tracking](https://ezpz.cool/configuration/#experiment-tracking)).
`--report-to none` alone is not enough: it only silences HF's own Trainer,
while ezpz's tracking is independent and defaults to `wandb`, so the first
validation run logged in and opened a run against the user's real project.
`EZPZ_TRACKER_BACKENDS=none` is the kill switch for *all* ezpz backends —
relevant because `mlflow` is also supported and auto-loads credentials from
`~/.amsc.env`, so disabling wandb alone would still leave a tracker able to
fire.

The `s`/`m`/`l` ladder is roughly 107M/248M/449M parameters at 400 iterations
(`test`). Measured on one Aurora node (12 ranks, `frameworks/2025.3.1`, ezpz
0.27.6): `test s` 48s / 107,147,274 params, `test m` 43s / 247,803,914 params,
`vit s` 46s, `fsdp s` 65s, `diffusion s` 91s, `hf` 72s, `hf_trainer` 42s. `m`
is not slower than `s` because its batch size halves (64 → 32) as the model
grows. `-test-m` and `-test-l` carry the `slow` tag. Note the whole
`benchmark` suite is opt-in and never runs by default.

A version caveat worth repeating here: installing ezpz with `--no-deps` leaves
its pins unenforced, and a stray `plotext` 6.x breaks **every** example with
`AttributeError: module 'plotext' has no attribute 'plot_size'` (ezpz pins
`plotext>=5,<6`; 6.0.0 is a rewrite that drops that API). The cases declare
`plotext` and `torchinfo` so a broken environment skips rather than reporting
a spurious benchmark failure.

## Retained manual sources

### ezpz

ezpz is **not part of the Frameworks SDK**. No `frameworks` module ships it, so
the copy that gets imported is whatever the user installed (user site,
`PYTHONUSERBASE`, or an active venv), and its version floats independently of
the SDK. That is precisely why the pairing is worth testing: the registered
cases validate *this* SDK against *the ezpz on the path*, and they skip cleanly
when no ezpz is installed rather than failing an SDK acceptance run over a
package the SDK does not provide.

Because the version floats, check it before filing a failure — a stale
user-site install reproduces bugs that were fixed upstream long ago.
`ezpz-environment` prints `ezpz=<version>` and `ezpz_path=[...]` first for
exactly this reason. These cases were validated against ezpz 0.27.6; upgrade
before investigating a failure:

```bash
python -m pip install --user --upgrade git+https://github.com/saforem2/ezpz
```

Two properties of ezpz shape how these tests are registered:

- Importing `ezpz` is cheap and MPI-free, but resolving rank/world (and
  therefore `setup_torch()`) reaches `mpi4py.MPI`, which aborts the interpreter
  outside an allocation: `Fatal error in internal_Init_thread`. The distributed
  cases consequently declare `min_xpus`, so they skip on a login node instead of
  producing a confusing abort.
- ezpz's lazy `__getattr__` turns a missing optional dependency into a missing
  *attribute* rather than an ImportError. `ezpz-environment` resolves each
  attribute the suite relies on so that failure mode surfaces as a named check
  instead of an AttributeError deep inside an application.

`ezpz-environment` (single process, no MPI) checks that ezpz agrees with the
loaded PyTorch about the accelerator: an ezpz reporting `cpu`/`gloo` on a
12-XPU node silently runs every ezpz-launched job on the CPU, and no
import-only test detects that.

`ezpz-distributed-{mpiexec,torchrun}` and `ezpz-launch` check bring-up results
rather than the absence of an exception: ranks agree with the launcher's own
environment, the gathered ranks are a permutation of `range(world_size)`, local
ranks map to *distinct* devices on each host, and an all-reduce over ezpz's
process group returns the closed-form answer. The distinct-device check is the
valuable one — every rank binding device 0 neither raises nor hangs.

It was verified non-vacuous by forcing `LOCAL_RANK=0` on every rank while all
12 XPUs stay visible; the test must fail with
`ranks on one host share device indices [0, 0, 0, 0]`. Note that
`ZE_AFFINITY_MASK=0` is *not* a valid injection against current ezpz: 0.27.6
rejects it inside `setup_torch()` with `RuntimeError: The device index is out
of range`, which is better ezpz behavior but aborts before the check runs, so
it no longer probes what it appears to.

All three launchers are registered because they drive genuinely different ezpz
code paths, and one passing does not imply the others:

- `mpiexec`/PALS: ezpz reads `PALS_*` rank variables.
- `torchrun`: ezpz reads `RANK`/`LOCAL_RANK` instead.
- `ezpz launch`: ezpz *constructs* the launch itself — it resolves the PBS
  hostfile, computes the rank geometry, builds the CPU-binding and `mpiexec`
  flags, and only then execs the payload. This is the path most ALCF jobs
  actually take, so an SDK change that breaks ezpz's hostfile parsing or
  binding logic is invisible to the other two cases.

`ezpz-launch` passes explicit `--nproc/--nproc_per_node` so the case is
reproducible at any allocation size. Dropping those flags makes ezpz derive the
full job geometry from PBS on its own, which is worth running manually on a
multi-node allocation when validating a new SDK:

```bash
ezpz launch -- python tests/distributed/ezpz_distributed.py
```

To run the registered cases directly:

```bash
mpiexec -n 12 -ppn 12 python tests/distributed/ezpz_distributed.py
torchrun --standalone --nproc-per-node=2 tests/distributed/ezpz_distributed.py
ezpz launch --nproc 2 --nproc_per_node 2 -- python tests/distributed/ezpz_distributed.py
TORCH_DEVICE=cpu TORCH_BACKEND=gloo python tests/smoke/ezpz_env.py
```

### JAX QMC

The QMC application writes a log and model and can run much longer than an
acceptance test. Run it only after reviewing `4He.ini`:

```bash
cd tests/workloads/jax_qmc
python Nuclear_ML.py 4He.ini
```

### vLLM offline inference

`tests/workloads/vllm/offline_inference.py` requires a model argument supported
by the installed vLLM version. The model path/ID and storage policy are site
decisions, so the harness does not invent one:

```bash
python tests/workloads/vllm/offline_inference.py --model MODEL_OR_PATH
```

The model-registry subprocess crash has a model-free registered regression.

### ML communication experiments

`tests/workloads/ml_communications/in_context_benchmarks/` retains the unique
Python implementations for 2D, GNN, LLM, and ViT communication experiments.
The source directory had dozens of run scripts tied to old module versions,
personal paths, queues, and output locations. Compose a new PBS job around the
desired payload rather than reviving those wrappers. The current scalable
collective suite should be the first correctness gate. The active XPU
sequence-parallel compute pattern was rewritten as the bounded registered
`workload-sequence-parallelism` case; the CUDA/XPU copies it superseded and a
near-identical A2A/HSN debug duplicate were consolidated.

### MoE drivers

The complete standalone driver set is retained beside the registered pytest
cases: `train.py`, `train_2d.py`, `train_pipeline.py`, and `generate.py`.
The harness registers one small, single-XPU `train.py` probe with finite-loss,
finite-gradient, and parameter-update checks. The 2-D, pipeline, checkpoint,
and generation choices remain manual because they require launch- and
model-specific decisions. To reproduce the registered training probe directly:

```bash
cd tests/workloads/moe
python train.py --arch k3 --steps 1 --batch-size 1 --seq-len 32 \
  --vocab 256 --dim 64 --layers 2 --experts 4 --top-k 2 \
  --device xpu --log-every 1
```

### Additional variants

- `tests/regressions/sdpa/repro_sdpa_dist.py` is the older ezpz/FSDP triage
  variant; the dependency-free GQA and full TP/FSDP variants are registered.
- `tests/workloads/gemm/matmul_from_torch_xpu_ops.py` is a broad shape/backward
  profiling sweep. The two latest forward GEMM comparisons are registered.
- `tests/workloads/torchcomms/allreduce.py` covers an alternate TorchComms API
  call shape and intentionally requires a multi-rank `torchrun` launch.
  `xccl_smoke.py` is the registered compatibility test, while
  `perf/collective_perf_test.py` provides the registered bounded adapter
  comparisons.
- `tests/workloads/moe/tools/` contains PMIx, P2P, determinism, and all-reduce
  diagnostic payloads for specialized distributed launches.
- `tests/regressions/ipex_pybind11/WA/` preserves the historical workaround
  patch for diagnosis; the registered reproducer tests the unmodified module.
