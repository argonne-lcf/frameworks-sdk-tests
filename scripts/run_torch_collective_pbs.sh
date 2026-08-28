#!/usr/bin/env bash
# Generic PBS launcher for tests/distributed/torch_collectives.
#
# Preferred submission form (the wrapper applies PROJECT, QUEUE, and node defaults):
#   TEST_CASE=allreduce PROJECT=datascience QUEUE=workq ./run_torch_collective_pbs.sh --submit
#
# It may also be submitted directly when scheduler options are supplied explicitly:
#   qsub -A datascience -q workq -l select=2 -v TEST_CASE=allreduce \
#       scripts/run_torch_collective_pbs.sh
#
# TEST_CASE is one of allreduce, allgather, alltoall, alltoall_uneven,
# reduce_scatter, overlap, p2p, streams, or subgroups. All TEST_* settings are inherited.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

find_repo_root() {
    local candidate
    for candidate in \
            "${REPO_ROOT:-}" \
            "${SCRIPT_DIR}/.." \
            "${PBS_O_WORKDIR:-}" \
            "${PBS_O_WORKDIR:-}/.."; do
        [[ -n "${candidate}" ]] || continue
        if [[ -f "${candidate}/tests/distributed/torch_collectives/_common.py" ]]; then
            (cd -- "${candidate}" && pwd)
            return 0
        fi
    done
    return 1
}

if ! REPO_ROOT=$(find_repo_root); then
    echo "cannot locate repository root; export REPO_ROOT before submission" >&2
    exit 2
fi
export REPO_ROOT

TEST_CASE=${TEST_CASE:-allreduce}
TEST_CASE=${TEST_CASE#test_torch_}
FRAMEWORKS_MODULE=${FRAMEWORKS_MODULE:-frameworks}
PROJECT=${PROJECT:-datascience}
QUEUE=${QUEUE:-workq}
FILESYSTEMS=${FILESYSTEMS:-home}
MASTER_PORT=${MASTER_PORT:-2345}
PYTHON=${PYTHON:-python}
XPUS_PER_NODE=${XPUS_PER_NODE:-12}

case "${TEST_CASE}" in
allreduce|allgather|alltoall|alltoall_uneven|reduce_scatter|overlap|p2p|subgroups)
    NNODES=${NNODES:-2}
    NRANKS_PER_NODE=${NRANKS_PER_NODE:-12}
    WALLTIME=${WALLTIME:-00:20:00}
    ;;
streams)
    # Streams is a local-device question, not a fabric test. One rank on one node gives
    # the cleanest signal; callers can raise either value to measure contention.
    NNODES=${NNODES:-1}
    NRANKS_PER_NODE=${NRANKS_PER_NODE:-1}
    WALLTIME=${WALLTIME:-00:15:00}
    ;;
*)
    echo "unknown TEST_CASE: ${TEST_CASE}" >&2
    exit 2
    ;;
esac

for value_name in NNODES NRANKS_PER_NODE MASTER_PORT XPUS_PER_NODE; do
    value=${!value_name}
    if [[ ! ${value} =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer (got ${value})" >&2
        exit 2
    fi
done
if (( MASTER_PORT > 65535 )); then
    echo "MASTER_PORT must be at most 65535 (got ${MASTER_PORT})" >&2
    exit 2
fi
if (( NRANKS_PER_NODE > XPUS_PER_NODE )); then
    echo "NRANKS_PER_NODE=${NRANKS_PER_NODE} exceeds XPUS_PER_NODE=${XPUS_PER_NODE}" >&2
    exit 2
fi

case "${1:-}" in
--submit)
    command -v qsub >/dev/null 2>&1 || {
        echo "qsub is not available" >&2
        exit 127
    }
    submit_args=(
        -V
        -A "${PROJECT}"
        -q "${QUEUE}"
        -l "select=${NNODES}"
        -l place=scatter
        -l "walltime=${WALLTIME}"
        -j oe
        -k doe
        -N "torch_${TEST_CASE}"
    )
    if [[ -n ${FILESYSTEMS} ]]; then
        submit_args+=(-l "filesystems=${FILESYSTEMS}")
    fi
    exec qsub "${submit_args[@]}" "${BASH_SOURCE[0]}"
    ;;
--help|-h)
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
"") ;;
*)
    echo "usage: $0 [--submit|--help]" >&2
    exit 2
    ;;
esac

if [[ -z ${PBS_NODEFILE:-} || ! -r ${PBS_NODEFILE} ]]; then
    echo "not running inside a PBS allocation; use --submit or qsub this script" >&2
    exit 2
fi
command -v mpiexec >/dev/null 2>&1 || {
    echo "mpiexec is required inside the PBS allocation" >&2
    exit 127
}

TEST_FILE="${REPO_ROOT}/tests/distributed/torch_collectives/test_torch_${TEST_CASE}.py"
[[ -f ${TEST_FILE} ]] || {
    echo "test payload not found: ${TEST_FILE}" >&2
    exit 2
}

# The allocation is authoritative if PBS supplied a different node count than the
# submission default. PBS nodefiles may repeat hosts, so count unique names.
NNODES=$(sort -u -- "${PBS_NODEFILE}" | wc -l)
PALS_WORLD_SIZE=$((NNODES * NRANKS_PER_NODE))
export PALS_WORLD_SIZE

# Lmod's exported shell function probes variables for several shells and is not safe
# under bash nounset on all site images. Limit the relaxation to module initialization.
set +u
module load "${FRAMEWORKS_MODULE}"
set -u

# Preserve the configuration users actually run while keeping the original suite's
# accelerator defaults. Every value remains overrideable at submission time.
export CCL_OP_SYNC=${CCL_OP_SYNC:-0}
export CCL_ATL_SYNC_COLL=${CCL_ATL_SYNC_COLL:-0}
export CCL_PROCESS_LAUNCHER=${CCL_PROCESS_LAUNCHER:-pmix}
export CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT:-mpi}
export ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}
export FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:-userfaultfd}
export CCL_WORKER_AFFINITY=${CCL_WORKER_AFFINITY:-42,43,44,45,46,47,94,95,96,97,98,99}
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1,2,3,4,5,6,7,8,9,10,11}

export TEST_DTYPE=${TEST_DTYPE:-bfloat16}
export TEST_TIMEOUT=${TEST_TIMEOUT:-600}
if [[ ${TEST_CASE} == streams ]]; then
    export TEST_GEMM_M=${TEST_GEMM_M:-8192}
    export TEST_GEMM_REPS=${TEST_GEMM_REPS:-20}
    export TEST_COPY_GB=${TEST_COPY_GB:-4}
else
    export TEST_MEM_BUDGET_GB=${TEST_MEM_BUDGET_GB:-50}
    export TEST_ITERS=${TEST_ITERS:-10}
fi

case "${TEST_CASE}" in
alltoall_uneven)
    export TEST_SKEW=${TEST_SKEW:-skew}
    export TEST_HOT=${TEST_HOT:-8}
    ;;
overlap)
    # CCL_OP_SYNC=1 forces synchronous completion and defeats this test.
    export TEST_OVERLAP_COLL=${TEST_OVERLAP_COLL:-all_reduce}
    export TEST_COMM_CALIB=${TEST_COMM_CALIB:-12}
    ;;
p2p)
    export TEST_P2P=${TEST_P2P:-ring}
    # Cross the fabric by default instead of keeping most adjacent hops on Xe Link.
    export TEST_P2P_STRIDE=${TEST_P2P_STRIDE:-${NRANKS_PER_NODE}}
    ;;
subgroups)
    export TEST_GROUPS=${TEST_GROUPS:-ep}
    export TEST_GROUP_CALIB=${TEST_GROUP_CALIB:-8}
    ;;
esac

bind_args=("${NRANKS_PER_NODE}" "${CPU_BIND_SHIFT:-0}")
if [[ ${CPU_BIND_LOGICAL:-0} == 1 ]]; then
    bind_args+=(--logical)
fi
if ! cpu_affinity_output=$(bash "${REPO_ROOT}/scripts/get_cpu_bind_aurora.sh" "${bind_args[@]}"); then
    echo "could not derive Aurora CPU affinity" >&2
    exit 2
fi
read -r -a cpu_affinity <<<"${cpu_affinity_output}"

MASTER_ADDR=${MASTER_ADDR:-$(head -n 1 "${PBS_NODEFILE}")}
export MASTER_ADDR MASTER_PORT

echo "TEST_CASE=${TEST_CASE} NNODES=${NNODES} NRANKS_PER_NODE=${NRANKS_PER_NODE} WORLD_SIZE=${PALS_WORLD_SIZE}"
echo "FRAMEWORKS_MODULE=${FRAMEWORKS_MODULE} TEST_FILE=${TEST_FILE}"

cd -- "${REPO_ROOT}"
exec mpiexec -n "${PALS_WORLD_SIZE}" -ppn "${NRANKS_PER_NODE}" -l --line-buffer \
    "${cpu_affinity[@]}" -env MASTER_ADDR="${MASTER_ADDR}" -env MASTER_PORT="${MASTER_PORT}" \
    "${PYTHON}" "${TEST_FILE}"
