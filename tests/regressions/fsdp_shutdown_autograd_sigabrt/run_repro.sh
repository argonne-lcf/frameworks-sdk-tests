#!/usr/bin/env bash
# Harness entry point for the FSDP autograd interpreter-shutdown SIGABRT
# reproducer (argonne-lcf/AuroraBugTracking#182).
#
# With default CPU binding and the libfabric MR-cache monitor disabled, an
# autograd worker thread releases queued Python callback references while the
# main thread is inside Py_FinalizeEx, ending in std::terminate/SIGABRT.
# Disabling MR-cache monitoring prevents the separate memhooks SIGSEGV from
# masking this failure.
#
# The workload itself must succeed on every rank and print its final
# "work done <rank>" line. mpiexec's exit status is the test result. The
# launch pins the core rlimit to zero (with PALS_RLIMITS=CORE) so crashes leave
# no core files behind in the run directory.
set -euo pipefail

if (( $# != 0 )); then
    echo "usage: $0" >&2
    exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The harness loads the frameworks module and captures its environment
# before invoking this script. When submitted directly with qsub, load it
# here instead.
if [[ "${LOADEDMODULES:-}" != *frameworks* ]]; then
    if ! command -v module >/dev/null 2>&1; then
        echo "frameworks module is not loaded and no module command is available;" >&2
        echo "load it first or run through the harness (./run_tests run ...)" >&2
        exit 2
    fi
    set +u
    module load "${FRAMEWORKS_MODULE:-frameworks}"
    set -u
fi

if [[ -z "${PBS_NODEFILE:-}" || ! -r "${PBS_NODEFILE}" ]]; then
    echo "not running inside a PBS allocation; run inside one or submit with qsub" >&2
    echo "(see README.md)" >&2
    exit 2
fi
command -v mpiexec >/dev/null 2>&1 || {
    echo "mpiexec is required inside the PBS allocation" >&2
    exit 127
}

PYTHON="${PYTHON:-python}"
NRANKS_PER_NODE="${NRANKS_PER_NODE:-12}"
MASTER_PORT="${MASTER_PORT:-30032}"
MPIEXEC_TIMEOUT="${MPIEXEC_TIMEOUT:-600}"

if [[ ! "${NRANKS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NRANKS_PER_NODE must be a positive integer (got ${NRANKS_PER_NODE})" >&2
    exit 2
fi

# PBS nodefiles may repeat hosts; count unique names.
NNODES=$(sort -u -- "${PBS_NODEFILE}" | wc -l)
WORLD_SIZE=$((NNODES * NRANKS_PER_NODE))
MASTER_ADDR="${MASTER_ADDR:-$(head -n 1 "${PBS_NODEFILE}")}"

export WORLD_SIZE MASTER_ADDR MASTER_PORT
# Propagate the caller's (zero) core rlimit to every rank.
ulimit -c 0
export PALS_RLIMITS="${PALS_RLIMITS:-CORE}"

# Disable MR-cache monitoring so the #180 SIGSEGV cannot mask this abort.
export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-disabled}"

# Retain mpiexec's default CPU binding for the #182 reproduction conditions.
mpiexec_args=(-n "${WORLD_SIZE}" -ppn "${NRANKS_PER_NODE}" -l --line-buffer)
if [[ "${MPIEXEC_TIMEOUT}" != "0" ]]; then
    mpiexec_args+=(--timeout "${MPIEXEC_TIMEOUT}")
fi

echo "FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR}"
echo "NNODES=${NNODES} NRANKS_PER_NODE=${NRANKS_PER_NODE} WORLD_SIZE=${WORLD_SIZE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} MPIEXEC_TIMEOUT=${MPIEXEC_TIMEOUT}"

exec mpiexec "${mpiexec_args[@]}" \
    "${PYTHON}" -u "${HERE}/repro_fsdp_shutdown.py"
