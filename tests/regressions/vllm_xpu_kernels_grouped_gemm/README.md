# XE2 grouped-GEMM D-store reproducer

This standalone SYCL test exercises the native XE2 output-store path used by
the FP16 grouped-GEMM kernel. It initializes a 4x512 output with NaNs, stores
one to every element, and fails if any element is missing or incorrect.

## What it reproduces

The XE2 D-store path in vLLM XPU kernels can leave part of the output tile
unwritten. The reproducer fills the output with NaNs and requires every
element to be overwritten with one.

## How to run

Through the harness (the case fails while the bug is present):

```bash
./run_tests run --id regression-vllm-xpu-kernels-grouped-gemm --module frameworks
```

Directly (builds in a temporary directory):

```bash
./run_repro.sh
```

`GROUPED_GEMM_WORK_DIR` reuses an existing checkout/build directory:

```bash
GROUPED_GEMM_WORK_DIR=/tmp/grouped-gemm ./run_repro.sh
```

The build clones `vllm-project/vllm-xpu-kernels` and downloads the Intel
SYCL-TLA headers, so it needs network access (`HTTPS_PROXY` when required).
`VLLM_CUTLASS_SRC_DIR` overrides the SYCL-TLA checkout.

## Expected result

The binary exits nonzero when the store path is broken and prints the NaN
count and affected output rows; the harness reports the case as failed. It
exits zero once every element is written. On the affected stack the exit
status is `1`; with the Intel AICOE GPU UMD module
(`module load intel_gpu_umd_aicoe/2026.06.19`) it is `0`.

## Files

- `run_repro.sh`: harness entry point; builds in a work directory and runs the
  binary.
- `build_repro_xe2_store_sycl.sh`: fetches sources and compiles the standalone
  test with `icpx`.
- `repro_xe2_store_sycl.cpp`: the standalone SYCL reproducer.
