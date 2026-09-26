#!/usr/bin/env bash
# Wrapper around `ezpz benchmark` for the frameworks SDK suite.
#
# Two things this wrapper exists to do, neither of which the manifest can
# express on its own:
#
#   1. Route output into the harness's per-case artifact directory. Test
#      commands are argv arrays and are never templated, so the artifact path
#      can only be picked up from FRAMEWORKS_TEST_ARTIFACT_DIR at runtime.
#      Without it, `ezpz benchmark` writes into ./outputs/ in the repo.
#
#   2. Refuse to launch an unbounded example. `ezpz benchmark` passes only
#      `--model` to each example, and the examples do not share a bounding
#      flag:
#
#        test       --train-iters   (preset-driven, 400 at s/m/l)
#        vit        --max-iters     (default 224)
#        fsdp       --epochs        (default 10)
#        fsdp_tp    --epochs        (default 5, over the full imdb dataset)
#        diffusion  (no iteration cap at all)
#
#      The epoch-based examples are dataset-scaled, not step-capped, so their
#      runtime is set by the corpus rather than by any flag `ezpz benchmark`
#      forwards. Observed on Aurora at --model s, 12 ranks: `test` finished in
#      50s and `fsdp` in 44s, while `fsdp_tp` was still running when a 30
#      minute job wall killed it, producing no timing row at all.
#
#      So this wrapper allows only examples that are genuinely bounded at a
#      known cost, and tells you to use a dedicated job for the rest rather
#      than letting an acceptance run hang until the scheduler kills it.
set -uo pipefail

EXAMPLE=${1:-test}
MODEL=${2:-s}

# Bounded by construction: iteration-capped, no dataset-scaled epoch loop.
case "${EXAMPLE}" in
test | vit) ;;
fsdp | fsdp_tp | diffusion)
    echo "refusing to run '${EXAMPLE}' from the test suite: it is bounded by" >&2
    echo "--epochs over a full dataset (or not bounded at all), so its runtime" >&2
    echo "is set by the corpus, not by anything 'ezpz benchmark' forwards." >&2
    echo "Run it in a dedicated job with an explicit budget instead, e.g.:" >&2
    echo "  ezpz launch -- python3 -m ezpz.examples.${EXAMPLE} --model ${MODEL} --epochs 1" >&2
    exit 2
    ;;
*)
    echo "unknown or unsupported example: ${EXAMPLE}" >&2
    exit 2
    ;;
esac

OUTDIR=${FRAMEWORKS_TEST_ARTIFACT_DIR:-${PWD}/outputs/ezpz-benchmarks}
mkdir -p -- "${OUTDIR}"

command -v ezpz >/dev/null 2>&1 || {
    echo "ezpz executable not found on PATH" >&2
    exit 127
}

echo "ezpz benchmark: example=${EXAMPLE} model=${MODEL} outdir=${OUTDIR}"
ezpz benchmark --run "${EXAMPLE}" --model "${MODEL}" --outdir "${OUTDIR}"
rc=$?

# `ezpz benchmark` already exits nonzero if any example failed; surface the
# machine-readable timings next to the harness log either way.
if [[ -f "${OUTDIR}/timings.csv" ]]; then
    echo "--- timings.csv ---"
    cat -- "${OUTDIR}/timings.csv"
else
    echo "warning: no timings.csv written to ${OUTDIR}" >&2
    # No timings row means the example never completed -- treat a "success"
    # with no measurement as a failure rather than a silent pass.
    if [[ ${rc} -eq 0 ]]; then
        echo "benchmark reported success but produced no timings; failing" >&2
        rc=1
    fi
fi

exit "${rc}"
