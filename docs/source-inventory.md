# Source inventory and migration record

Migration date: 2026-08-28. The source tree at
`/lus/tegu/projects/datasets/software/testing` was read only; no source files
were changed.

## Sources retained

| Source | Provenance | What was retained and normalized |
| --- | --- | --- |
| `fw_sdk_main/frameworks-sdk` | branch `users/khalid/test-suite-pytorch`, commit `f72aa6e` | The current shared validator and eight PyTorch collective payloads. Repeated PBS, affinity, and vLLM helper copies were replaced by one PBS launcher and one affinity helper. |
| `huihuo_testing_frameworks/test_frameworks` | branch `users/khalid/cicd`, commit `ea2adce`, dirty working tree | Unique 1-D/2-D DTensor, GPU-aware mpi4py, checkpoint, and model-training intent. Byte-identical root/nested copies and obsolete collective implementations were removed. Workloads were bounded, made synthetic by default, and given correctness checks. |
| `deepspeed_jit_test`, `dpctl_ray_bug`, `dpnp_test`, `gamma_dist_bug`, `sdpa_stride_bug`, `jaehuyk_ddp_test` | current filesystem versions | Focused test intent was extracted. Personal environment wrappers were removed; missing hardware is no longer treated as success; output-only examples now assert results. |
| `ipex_jit_compile_bug/frameworks-sdk` | stale nested SDK checkout | Only the canonical, newer IPEX pybind11 reproducer from `fw_sdk_main` was retained. Generated copies of installed IPEX source were excluded. |
| `standalone_moe_example` | current filesystem version | `latentmoe`, all four Python training/generation drivers, its pytest suite, configuration, and communication smoke tools. Hard-coded module launchers, model checkpoints, and training outputs were excluded. |
| `torchcomm_example` | current filesystem version; `torchcomms-perf-alcf` branch `master`, commit `5d54c03` | Two TorchComms/XCCL correctness payloads and the portable installed-package collective perf payload. Local-rank device binding was moved ahead of communicator initialization and swallowed benchmark failures were removed. The dated duplicate, `.upstream` source mirror, site/PBS wrappers, logs, and results were excluded. |
| `vllm-efforts` | current filesystem version | Model-registry inspection and offline-inference source. Caches, logs, profiles, core dumps, and cluster setup experiments were excluded. |
| `jax_qmc_02_04_2026/JAX_QMC` | commit `f895eb2` plus current modified/untracked test inputs | Application source and small required model/potential inputs. Its hard-coded Conda/PBS launchers and generated log were excluded. |
| `ml_communications` | clean commit `8a3ac75` | In-context benchmark Python sources plus a bounded, correctness-aware XPU sequence-parallel case. A near-identical A2A/HSN pair and the CUDA/XPU sequence copies were consolidated; site/version-specific run scripts were excluded. |
| `cosmictagger_sow` | current filesystem version | Application source, dependency-split pytest cases, and the 3.5 MiB light fixture. Synthetic allocation/index bugs and current PyTorch/Keras incompatibilities were repaired; notebooks, analysis output, Git metadata, and legacy submission scripts were excluded. |
| `gemm_tests` | current filesystem version | Latest Python benchmark variants only. Older near-duplicates, profiler traces, binaries, results, XPU-SMI dumps, and the nested oneDNN checkout/build were excluded. |

The migration uses the newer `fw_sdk_main` collective suite as the source of
truth. The older `test_frameworks` collective scripts had no reliable failure
oracle (including an async broadcast that was never waited and an all-gather
timing array that was never populated), while the newer suite performs
deterministic cross-rank validation, hang detection, and reduced failure
propagation.

## Deliberately not copied

The following are source repositories, generated state, duplicates, or
investigation artifacts—not portable Frameworks SDK acceptance tests:

| Source item | Approximate size | Reason |
| --- | ---: | --- |
| `resnet50/resnet50_traces`, CIFAR data, checkpoint | 14 GiB total | Generated profiles/data/model; one cleaned synthetic ResNet workload is retained instead. |
| `sdpa_stride_bug/{torchinductor_cache,venvs,logs,outputs}` | 755 MiB total | Generated compiler cache, environment, and output; Python reproducers are retained. |
| `vllm-efforts` core dumps and caches | multiple GiB | Generated crash/cache state; focused source repro is retained. |
| `torch_2.13.0_unit_tests/pytorch` | 1.67 GiB | Full upstream PyTorch checkout, not a local SDK test. |
| `vllm_xpu_kernels_tests/vllm-xpu-kernels` | 1.77 GiB | Full upstream source/test checkout, including a 1.9 GiB core dump. Installed-package import is covered separately. |
| `triton_tests/intel-xpu-backend-for-triton` | 275 MiB | Full upstream compiler checkout. Installed Triton import and MoE kernel tests cover the SDK artifact. |
| `deepspeed_examples/DeepSpeedExamples` | 208 MiB | Upstream example checkout; a focused JIT test and bounded miniGPT workload are retained. |
| `gemm_tests/onednn_08_04_2026` | about 811 MiB | Upstream checkout/build rather than a Frameworks SDK test. |
| `torchcomm_example_old_08_25_2026` | dated duplicate | Superseded by the current TorchComms payloads. |
| historical `.conda.list` / `.pip.list` files | negligible | Package snapshots, not executable tests. |
| `module_bug/condash_path.log` | 0 bytes | Empty file; no executable test. |
| `vim_testing` | negligible | Scratch text only. |

No `.git` directories, `__pycache__`, `.pytest_cache`, virtual environments,
compiler caches, model checkpoints, trace JSON, scheduler output directories,
or personal token-loading wrappers were copied. In particular, the original
SDPA shell wrappers used `bash -x` while reading a Hugging Face token; those
wrappers were intentionally discarded to prevent credential expansion into
logs.
