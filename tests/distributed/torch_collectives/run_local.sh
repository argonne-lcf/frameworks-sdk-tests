#!/usr/bin/env bash
# Local CPU/gloo smoke test -- proves the harness before it reaches a compute system.
# SINGLE NODE ONLY, purely a development convenience. The production launch path is
# mpiexec + PALS via scripts/run_torch_collective_pbs.sh.
#
#   ./run_local.sh        all 9 test payloads / 18 execution modes; every case must PASS
#   ./run_local.sh pals   all 9 payloads with rank/world resolved from PALS_* only
#   ./run_local.sh neg    fault injection; every case must be DETECTED
#
# Override: RESULTS_DIR, TORCHRUN, PYTHON, NP, PORT, TEST_MEM_BUDGET_GB,
# TEST_ITERS, TEST_DTYPE, or any payload-specific TEST_* variable.
set -uo pipefail

SUITE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RESULTS_DIR=${RESULTS_DIR:-${FRAMEWORKS_TEST_ARTIFACT_DIR:-"${SUITE_DIR}/results"}}
TORCHRUN=${TORCHRUN:-torchrun}
PYTHON=${PYTHON:-python}
NP=${NP:-4}
PORT=${PORT:-${MASTER_PORT:-29580}}

mkdir -p -- "${RESULTS_DIR}"

tests=(
    allreduce
    allgather
    alltoall
    alltoall_uneven
    reduce_scatter
    overlap
    p2p
    streams
    subgroups
)
cases=(
    allreduce:
    allgather:
    alltoall:
    alltoall_uneven:
    reduce_scatter:
    overlap:all_reduce
    overlap:all_gather
    p2p:ring
    p2p:ring_async
    p2p:ring_async_sym
    p2p:ring_batch
    p2p:batch
    streams:
    subgroups:ep
    subgroups:pp
    subgroups:seq
    subgroups:disjoint
    subgroups:overlap
)

export TEST_DEVICE=cpu
export TEST_MEM_BUDGET_GB=${TEST_MEM_BUDGET_GB:-0.02}
export TEST_ITERS=${TEST_ITERS:-3}
rc=0

summary() {
    local logfile=$1
    grep -h -m1 -E 'local check|no accelerator streams|RESULT PASS' "${logfile}" || true
}

case "${1:-run}" in
neg)
    command -v "${TORCHRUN}" >/dev/null 2>&1 || {
        echo "torchrun executable not found: ${TORCHRUN}" >&2
        exit 127
    }
    # Small chunks ensure smoke-test buffers straddle RNG chunk boundaries.
    for mode in misroute nan scale offset p2p groups; do
        logfile="${RESULTS_DIR}/neg_${mode}.log"
        if NEG="${mode}" TEST_CHUNK=${TEST_CHUNK:-4096} \
                "${TORCHRUN}" --nproc_per_node="${NP}" --master_port="${PORT}" \
                "${SUITE_DIR}/_negtest.py" >"${logfile}" 2>&1; then
            printf 'BAD   %-22s NOT detected -- see %s\n' "${mode}" "${logfile}"
            rc=1
        elif ! grep -q 'RESULT FAIL' "${logfile}"; then
            printf 'FAIL  %-22s harness error, not expected detection -- see %s\n' \
                "${mode}" "${logfile}"
            rc=1
        else
            printf 'ok    %-22s detected (%s)\n' "${mode}" "${logfile}"
        fi
    done
    ;;
pals)
    command -v "${PYTHON}" >/dev/null 2>&1 || {
        echo "python executable not found: ${PYTHON}" >&2
        exit 127
    }
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=${PORT}
    printf 'Running %d payloads with PALS-style environment; logs: %s\n' \
        "${#tests[@]}" "${RESULTS_DIR}"
    for test_name in "${tests[@]}"; do
        pids=()
        for ((rank = 0; rank < NP; rank++)); do
            logfile="${RESULTS_DIR}/pals_${test_name}_${rank}.log"
            env -u RANK -u WORLD_SIZE -u LOCAL_RANK \
                PALS_RANKID="${rank}" PALS_WORLD_SIZE="${NP}" \
                PALS_LOCAL_RANKID="${rank}" PALS_LOCAL_SIZE="${NP}" \
                "${PYTHON}" "${SUITE_DIR}/test_torch_${test_name}.py" \
                >"${logfile}" 2>&1 &
            pids+=("$!")
        done

        failed=0
        for pid in "${pids[@]}"; do
            wait "${pid}" || failed=1
        done
        if ((failed == 0)); then
            printf 'ok    %-22s %s\n' "${test_name}" \
                "$(summary "${RESULTS_DIR}/pals_${test_name}_0.log")"
        else
            printf 'FAIL  %-22s see %s/pals_%s_*.log\n' \
                "${test_name}" "${RESULTS_DIR}" "${test_name}"
            rc=1
        fi
    done
    ;;
run)
    command -v "${TORCHRUN}" >/dev/null 2>&1 || {
        echo "torchrun executable not found: ${TORCHRUN}" >&2
        exit 127
    }
    printf 'Running %d test payloads across %d modes; logs: %s\n' \
        "${#tests[@]}" "${#cases[@]}" "${RESULTS_DIR}"
    # gloo cannot reproduce the accelerator stream-ordering deadlock in
    # ring_async_sym, but the local run still proves that every mode dispatches.
    for case_spec in "${cases[@]}"; do
        IFS=: read -r test_name mode <<<"${case_spec}"
        label=${test_name}${mode:+.${mode}}
        mode_env=()
        case "${test_name}" in
        overlap) mode_env+=("TEST_OVERLAP_COLL=${mode}") ;;
        p2p) mode_env+=("TEST_P2P=${mode}") ;;
        subgroups) mode_env+=("TEST_GROUPS=${mode}") ;;
        esac

        logfile="${RESULTS_DIR}/run_${label}.log"
        if env "${mode_env[@]}" "${TORCHRUN}" --nproc_per_node="${NP}" \
                --master_port="${PORT}" "${SUITE_DIR}/test_torch_${test_name}.py" \
                >"${logfile}" 2>&1; then
            printf 'ok    %-22s %s\n' "${label}" "$(summary "${logfile}")"
        else
            printf 'FAIL  %-22s see %s\n' "${label}" "${logfile}"
            rc=1
        fi
    done
    ;;
*)
    echo "usage: $0 [run|pals|neg]" >&2
    exit 2
    ;;
esac

if ((rc == 0)); then
    echo "ALL OK"
else
    echo "FAILURES"
fi
exit "${rc}"
