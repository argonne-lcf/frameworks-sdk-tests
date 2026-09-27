#!/usr/bin/env bash
# Wrapper around the ezpz example benchmarks for the frameworks SDK suite.
#
# Three things this wrapper exists to do, none of which the manifest can
# express on its own:
#
#   1. Route output into the harness's per-case artifact directory. Test
#      commands are argv arrays and are never templated, so the artifact path
#      can only be picked up from FRAMEWORKS_TEST_ARTIFACT_DIR at runtime.
#      Without it, `ezpz benchmark` writes into ./outputs/ in the repo.
#
#   2. Bound the examples that `ezpz benchmark` cannot bound itself. It
#      forwards only `--model`, and the examples do not share a bounding flag,
#      so each was measured at its `ezpz benchmark` defaults (one Aurora node,
#      12 ranks, --model s):
#
#        test       --train-iters (400 at s/m/l)   48s
#        vit        --max-iters (default 224)      46s
#        fsdp       --epochs 10 over MNIST         65s
#        diffusion  --train-steps (default 400)    91s
#        fsdp_tp    --epochs 5                     TIMEOUT (>600s on any dataset)
#
#      fsdp_tp is the one that needs help: at --epochs 5 it exceeds 600s on
#      imdb *and* on mnist, so the corpus is not the driver -- the epoch count
#      is. Capped at --epochs 1 it completes (419s imdb / 537s random), so it
#      is registered through a direct launch with an explicit budget rather
#      than refused.
#
#   3. Run the HuggingFace examples off a public model with an explicit step
#      budget. `ezpz benchmark --run hf` hard-codes meta-llama/Llama-3.2-1B
#      (gated: an anonymous fetch of its config returns 401, versus 307 for
#      Qwen) and `--report-to=wandb`. Neither belongs in an SDK acceptance
#      run, and the model choice is not overridable through `ezpz benchmark`,
#      so those two cases launch the example module directly and synthesize
#      the same timings.csv contract.
set -uo pipefail

EXAMPLE=${1:-test}
MODEL=${2:-s}

# Validate the example BEFORE checking for ezpz: rejecting an unknown example
# is a property of the request, not of the environment, so it must report the
# same way whether or not ezpz happens to be installed.
case "${EXAMPLE}" in
test | vit | fsdp | diffusion | fsdp_tp | hf | hf_trainer) ;;
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

# Public, ungated, and small enough to fine-tune briefly on one node.
# Override with EZPZ_BENCH_HF_MODEL to benchmark a different checkpoint.
HF_MODEL=${EZPZ_BENCH_HF_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
HF_DATASET=${EZPZ_BENCH_HF_DATASET:-eliplutchok/fineweb-small-sample}
HF_MAX_STEPS=${EZPZ_BENCH_HF_MAX_STEPS:-20}
HF_BLOCK_SIZE=${EZPZ_BENCH_HF_BLOCK_SIZE:-1024}

# fsdp_tp's cost is driven by its epoch count, not its corpus: at the
# `ezpz benchmark` default of --epochs 5 it exceeds 600s on imdb and on mnist
# alike. One epoch completes (419s imdb / 537s random on one node).
FSDP_TP_DATASET=${EZPZ_BENCH_FSDP_TP_DATASET:-random}
FSDP_TP_EPOCHS=${EZPZ_BENCH_FSDP_TP_EPOCHS:-1}

run_hf_example() {
    # `ezpz benchmark` cannot express these overrides, so drive the module
    # directly. --max-steps is what makes this bounded; --report-to=none stops
    # HF's own Trainer from reporting.
    #
    # ezpz's tracking is separate from HF's report_to and has to be turned off
    # on its own. This is the offline recipe from the ezpz docs
    # (https://ezpz.cool/configuration/#experiment-tracking):
    # EZPZ_TRACKER_BACKENDS=none is the documented kill switch for *all*
    # backends -- ezpz supports wandb, csv, and mlflow, and mlflow
    # auto-loads credentials from ~/.amsc.env, so disabling wandb alone would
    # still leave a tracker able to fire. WANDB_DISABLED is kept alongside it
    # exactly as the docs pair them.
    local module=$1
    local t0=$SECONDS
    EZPZ_TRACKER_BACKENDS=none WANDB_DISABLED=1 \
        ezpz launch -- python3 -m "ezpz.examples.${module}" \
        --model_name_or_path "${HF_MODEL}" \
        --dataset_name "${HF_DATASET}" \
        --streaming \
        --bf16=true \
        --do_train=true \
        --do_eval=false \
        --max-steps "${HF_MAX_STEPS}" \
        --block_size "${HF_BLOCK_SIZE}" \
        --per_device_train_batch_size 1 \
        --logging-steps 1 \
        --logging-first-step \
        --optim adamw_torch \
        --report-to none \
        --overwrite_output_dir \
        --output_dir "${OUTDIR}/hf-output"
    local rc=$? el=$((SECONDS - t0))
    # Mirror the timings.csv that `ezpz benchmark` would have written, so every
    # registered benchmark case exposes the same machine-readable artifact.
    printf 'name,exit_code,wall_seconds\n%s,%d,%d\n' "${module}" "${rc}" "${el}" \
        >"${OUTDIR}/timings.csv"
    return "${rc}"
}

run_fsdp_tp() {
    # `ezpz benchmark` runs this at --epochs 5, which exceeds 600s on both
    # imdb and mnist, so it cannot be driven through `ezpz benchmark`. Launch
    # it directly with an explicit epoch budget and synthesize the same
    # timings.csv contract.
    local t0=$SECONDS
    EZPZ_TRACKER_BACKENDS=none WANDB_DISABLED=1 \
        ezpz launch -- python3 -m ezpz.examples.fsdp_tp \
        --model "${MODEL}" \
        --dataset "${FSDP_TP_DATASET}" \
        --epochs "${FSDP_TP_EPOCHS}"
    local rc=$? el=$((SECONDS - t0))
    printf 'name,exit_code,wall_seconds\nfsdp_tp,%d,%d\n' "${rc}" "${el}" \
        >"${OUTDIR}/timings.csv"
    return "${rc}"
}

case "${EXAMPLE}" in
hf | hf_trainer)
    echo "ezpz ${EXAMPLE}: model=${HF_MODEL} dataset=${HF_DATASET}" \
        "max_steps=${HF_MAX_STEPS} outdir=${OUTDIR}"
    run_hf_example "${EXAMPLE}"
    rc=$?
    ;;
fsdp_tp)
    echo "ezpz fsdp_tp: model=${MODEL} dataset=${FSDP_TP_DATASET}" \
        "epochs=${FSDP_TP_EPOCHS} outdir=${OUTDIR}"
    run_fsdp_tp
    rc=$?
    ;;
*)
    # test / vit / fsdp / diffusion: `ezpz benchmark` drives these correctly
    # and they are bounded at its defaults. Do NOT fall through to the HF
    # branch -- these parsers reject the HuggingFace TrainingArguments flags
    # with "unrecognized arguments" and exit 2.
    echo "ezpz benchmark: example=${EXAMPLE} model=${MODEL} outdir=${OUTDIR}"
    ezpz benchmark --run "${EXAMPLE}" --model "${MODEL}" --outdir "${OUTDIR}"
    rc=$?
    ;;
esac

# Surface the machine-readable timings next to the harness log.
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
