# Test catalog notes

`suite.json` is the authoritative executable catalog. See the README's
[test-selection guide](../README.md#selecting-tests) for suite guidance and
`./run_tests list` for registered test IDs, commands, and tags.

This document summarizes registered tests by suite and covers retained source
that needs an argument, model, scheduler, or investigation-specific decision
and therefore is not launched by a normal suite selection.

## Registered tests

Use `./run_tests list --suite SUITE` for executable commands and source paths.
The descriptions below focus on what each test validates or reproduces.

| Suite | Registered test | Purpose |
| --- | --- | --- |
| `smoke` | `imports-core` | Core package ABI and import surface |
| `smoke` | `pytorch-xpu` | PyTorch XPU and XCCL smoke validation |
| `smoke` | `mpi4py-single` | Single-process mpi4py and MPI environment validation |
| `smoke` | `dpctl-devices` | dpctl device discovery |
| `smoke` | `dpnp-array` | dpnp array creation and computation |
| `smoke` | `dpnp-torch-dlpack` | PyTorch/dpnp DLPack sharing |
| `optional-imports` | `imports-science` | NumPy, SciPy, h5py, scikit-learn, JAX, and TensorFlow imports |
| `optional-imports` | `imports-llm` | DeepSpeed, Transformers, Triton, vLLM, and vLLM XPU kernel imports |
| `optional-imports` | `imports-comms` | oneCCL PyTorch bindings and TorchComms imports |
| `optional-imports` | `imports-intel-extension` | Legacy Intel Extension for PyTorch import |
| `harness` | `collectives-cpu` | Collective-validator correctness on CPU/gloo |
| `harness` | `collectives-pals-env-cpu` | Collective validation through PALS-style environment variables |
| `harness` | `collectives-validator-negative` | Expected-fault injection and validator failure reporting |
| `distributed` | `torch-allreduce` | Correctness-aware two-XPU all-reduce |
| `distributed` | `torch-allgather` | Correctness-aware two-XPU all-gather |
| `distributed` | `torch-alltoall` | Correctness-aware two-XPU all-to-all |
| `distributed` | `torch-alltoall-uneven` | Correctness-aware uneven two-XPU all-to-all |
| `distributed` | `torch-reduce-scatter` | Correctness-aware two-XPU reduce-scatter |
| `distributed` | `torch-overlap-allreduce` | All-reduce and compute overlap |
| `distributed` | `torch-overlap-allgather` | All-gather and compute overlap |
| `distributed` | `torch-p2p-ring` | Ring point-to-point communication |
| `distributed` | `torch-p2p-ring-async` | Asynchronous ring point-to-point communication |
| `distributed` | `torch-p2p-ring-async-symmetric` | Symmetric asynchronous ring deadlock regression |
| `distributed` | `torch-p2p-ring-batch` | Batched ring point-to-point communication |
| `distributed` | `torch-p2p-batch` | Batched point-to-point communication |
| `distributed` | `torch-subgroups-ep` | Expert-parallel subgroup collectives on four XPUs |
| `distributed` | `torch-subgroups-pp` | Pipeline-parallel subgroup collectives on four XPUs |
| `distributed` | `torch-subgroups-sequential` | Sequential subgroup ordering |
| `distributed` | `torch-subgroups-disjoint` | Disjoint subgroup concurrency |
| `distributed` | `torch-subgroups-overlap` | Overlapping subgroup communication and ordering |
| `distributed` | `torch-streams` | Independent-stream copy/compute overlap |
| `distributed` | `mpi4py-xpu-allreduce` | Direct GPU-buffer `mpi4py.Allreduce` validation |
| `regression` | `regression-sdpa-gqa-stride` | GQA SDPA compiler crash |
| `regression` | `regression-sdpa-non-gqa-baseline` | Non-GQA SDPA baseline |
| `regression` | `regression-sdpa-tp-fsdp` | Full TP/FSDP SDPA reproducer |
| `regression` | `regression-gamma-large-sample` | Large Gamma-sample validation |
| `regression` | `regression-deepspeed-fused-adam-jit` | DeepSpeed FusedAdam JIT build |
| `regression` | `regression-ipex-pybind11-jit` | IPEX pybind11 JIT build |
| `regression` | `regression-vllm-model-inspection` | vLLM model-registry inspection and subprocess crash |
| `regression` | `regression-vllm-xpu-kernels-grouped-gemm` | XE2 grouped-GEMM D-store failure |
| `regression` | `regression-xccl-allgather-empty-cache` | XCCL all-gather `empty_cache` memory leak |
| `regression` | `regression-fsdp-shutdown-libfabric-memhooks` | FSDP libfabric memhooks shutdown SIGSEGV ([#180](https://github.com/argonne-lcf/AuroraBugTracking/issues/180)) |
| `regression` | `regression-fsdp-shutdown-autograd-sigabrt` | FSDP autograd callback SIGABRT during interpreter finalization ([#182](https://github.com/argonne-lcf/AuroraBugTracking/issues/182)) |
| `regression` | `regression-module-tracker-shutdown-sigabrt` | Single-process ModuleTracker SIGABRT during interpreter finalization ([#182](https://github.com/argonne-lcf/AuroraBugTracking/issues/182)) |
| `workload` | `workload-checkpoint-io` | MPI checkpoint I/O validation |
| `workload` | `workload-dtensor` | One-dimensional DTensor redistribution |
| `workload` | `workload-dtensor-2d-redistribute` | Two-dimensional DTensor-mesh redistribution |
| `workload` | `workload-mnist-ddp` | Bounded MNIST DDP training |
| `workload` | `workload-resnet50-ddp` | Bounded ResNet-50 DDP training |
| `workload` | `workload-resnet50-fsdp` | Bounded ResNet-50 FSDP training |
| `workload` | `workload-transformer-ddp` | Bounded Transformer DDP training |
| `workload` | `workload-minigpt-deepspeed` | Bounded DeepSpeed miniGPT training |
| `workload` | `workload-sequence-parallelism` | XPU sequence-parallel compute pattern |
| `workload` | `workload-torchcomms-xccl` | TorchComms/XCCL compatibility |
| `workload` | `workload-moe-reference-pytest` | MoE pytest cases against the CPU reference implementation |
| `workload` | `workload-moe-triton-xpu-pytest` | MoE Triton-required and XPU pytest cases |
| `workload` | `workload-moe-train-xpu` | Bounded single-XPU MoE training with loss, gradient, and parameter-update checks |
| `workload` | `workload-cosmictagger-io-synthetic` | CosmicTagger synthetic-data I/O tests |
| `workload` | `workload-cosmictagger-io-larcv` | CosmicTagger LArCV-data I/O tests |
| `workload` | `workload-cosmictagger-torch-synthetic` | CosmicTagger PyTorch synthetic-data training and inference |
| `workload` | `workload-cosmictagger-torch-larcv` | CosmicTagger PyTorch LArCV-data training and inference |
| `workload` | `workload-cosmictagger-tensorflow-synthetic` | CosmicTagger TensorFlow synthetic-data training and inference |
| `workload` | `workload-cosmictagger-tensorflow-larcv` | CosmicTagger TensorFlow LArCV-data training and inference |
| `workload` | `workload-cosmictagger-network-parity` | CosmicTagger network creation and PyTorch/TensorFlow parity |
| `benchmark` | `benchmark-torchcomms-perf` | Bounded two-rank TorchComms all-reduce measurements |
| `benchmark` | `benchmark-c10d-perf` | Bounded two-rank c10d all-reduce measurements |
| `benchmark` | `benchmark-c10d-torchcomms-perf` | Bounded two-rank c10d-through-TorchComms all-reduce measurements |
| `benchmark` | `benchmark-gemm-torch` | PyTorch forward GEMM comparison |
| `benchmark` | `benchmark-gemm-eleutherai` | EleutherAI forward GEMM comparison |

## Retained manual sources

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
