#!/usr/bin/env bash
# Harness entry point for the ModuleTracker interpreter-shutdown SIGABRT
# reproducer (argonne-lcf/AuroraBugTracking#182).
#
# The autograd callback-lifetime race during Py_FinalizeEx reproduced
# reliably with single-CPU affinity in the upstream report, so the payload
# runs under taskset pinned to one CPU from the current affinity mask. The
# workload must finish, print "work done", and the process must then exit
# zero; a SIGABRT during interpreter shutdown is a reproduction and makes
# this script exit nonzero. Core dumps are disabled so the abort path
# leaves nothing behind.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

command -v taskset >/dev/null 2>&1 || {
    echo "taskset is required to pin the reproducer to one CPU" >&2
    exit 127
}

ulimit -c 0

cpu="$("${PYTHON}" -c 'import os; print(min(os.sched_getaffinity(0)))')"
echo "pinning reproducer to CPU ${cpu}"

exec taskset -c "${cpu}" \
    "${PYTHON}" -u "${HERE}/repro_module_tracker_shutdown.py"
