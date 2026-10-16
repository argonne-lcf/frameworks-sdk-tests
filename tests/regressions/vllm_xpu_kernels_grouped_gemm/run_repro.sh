#!/usr/bin/env bash
# Harness entry point for the XE2 grouped-GEMM D-store reproducer.
#
# The build script fetches vllm-xpu-kernels and Intel SYCL-TLA into its
# current directory, so build in a work directory instead of the source tree.
# The binary exits nonzero when the XE2 D-store path drops or corrupts output
# elements; that status is the test result.
#
# GROUPED_GEMM_WORK_DIR keeps the checkout and build cache between runs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${GROUPED_GEMM_WORK_DIR:-}" ]]; then
  WORK_DIR="${GROUPED_GEMM_WORK_DIR}"
elif [[ -n "${FRAMEWORKS_TEST_ARTIFACT_DIR:-}" ]]; then
  WORK_DIR="${FRAMEWORKS_TEST_ARTIFACT_DIR}"
else
  WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/grouped-gemm-store.XXXXXX")"
  trap 'rm -rf "${WORK_DIR}"' EXIT
fi
mkdir -p "${WORK_DIR}"

cp "${HERE}/repro_xe2_store_sycl.cpp" "${HERE}/build_repro_xe2_store_sycl.sh" \
  "${WORK_DIR}/"
cd "${WORK_DIR}"

echo "Building XE2 grouped-GEMM D-store reproducer in ${WORK_DIR}"
bash ./build_repro_xe2_store_sycl.sh

echo "Running ${WORK_DIR}/repro_xe2_store_sycl"
if ./repro_xe2_store_sycl; then
  echo "PASS: every XE2 grouped-GEMM store element is correct"
else
  status=$?
  echo "FAIL: XE2 grouped-GEMM store dropped or corrupted output (exit ${status})" >&2
  exit "${status}"
fi
