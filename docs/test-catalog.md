# Test catalog notes

`suite.json` is the executable catalog. This document covers retained source
that needs an argument, model, scheduler, or investigation-specific decision
and therefore is not launched by a normal suite selection.

## Registered suites

- `smoke` checks the core package ABI/import surface, PyTorch XPU/XCCL,
  mpi4py, dpctl, dpnp, and PyTorch/dpnp DLPack sharing.
- `optional-imports` checks science, LLM, communication, and legacy Intel/IPEX
  package groups without making them part of the default acceptance gate.
- `harness` exercises the collective validator on CPU/gloo, through PALS-style
  environment variables, and with expected fault injection.
- `distributed` contains correctness-aware all-reduce, all-gather,
  all-to-all, uneven all-to-all, reduce-scatter, collective/compute overlap,
  five P2P modes, independent-stream overlap, five expert/pipeline/disjoint/
  overlapping subgroup communicator modes, and direct GPU-buffer
  `mpi4py.Allreduce` validation.
- `regression` contains the GQA SDPA compiler crash, its baseline, the full
  TP/FSDP SDPA reproducer, Gamma sampling, DeepSpeed and IPEX JIT builds, vLLM
  registry inspection, the XE2 grouped-GEMM D-store reproducer, and the XCCL
  `empty_cache` memory leak.
- `workload` contains checkpoint I/O, 1-D and 2-D DTensor redistribution,
  bounded MNIST/ResNet/Transformer training, DeepSpeed miniGPT, TorchComms,
  XPU sequence parallelism, separate MoE reference/Triton pytest gates, a
  bounded MoE training probe, and dependency-split CosmicTagger pytest cases.
- `benchmark` contains bounded two-rank all-reduce measurements for the
  TorchComms, c10d, and c10d-through-TorchComms adapters, plus GEMM sweeps.

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
