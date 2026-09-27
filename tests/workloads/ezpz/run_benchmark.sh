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
#   2. Refuse to launch the one example that is genuinely unbounded. `ezpz
#      benchmark` passes only `--model`, and the examples do not share a
#      bounding flag, so each was measured at the defaults `ezpz benchmark`
#      actually uses (one Aurora node, 12 ranks, --model s, 900s cap):
#
#        test       --train-iters (400 at s/m/l)   48s
#        vit        --max-iters (default 224)      46s
#        fsdp       --epochs 10 over MNIST         53s  (fixed-size corpus)
#        diffusion  --train-steps (default 400)    69s  (toy corpus default)
#        fsdp_tp    --epochs 5 over full imdb      TIMEOUT at 900s
#
#      Only fsdp_tp is unbounded: its cost scales with a 25k-row dataset at
#      seq_len 2048 and nothing `ezpz benchmark` forwards caps it. `--epochs`
#      alone is not the tell -- fsdp uses it too and finishes in under a
#      minute because MNIST is a fixed size.
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

# Validate the example BEFORE checking for ezpz: refusing an unbounded example
# is a property of the request, not of the environment, so it must report the
# same way whether or not ezpz happens to be installed.
case "${EXAMPLE}" in
test | vit | fsdp | diffusion | hf | hf_trainer) ;;
fsdp_tp)
    echo "refusing to run 'fsdp_tp' from the test suite: it trains for" >&2
    echo "--epochs (default 5) over the full imdb dataset at seq_len 2048," >&2
    echo "so its runtime is set by the corpus and nothing 'ezpz benchmark'" >&2
    echo "forwards can bound it. Measured: still running at a 900s cap." >&2
    echo "Run it in a dedicated job with an explicit budget instead, e.g.:" >&2
    echo "  ezpz launch -- python3 -m ezpz.examples.fsdp_tp --model ${MODEL} --epochs 1" >&2
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

# Public, ungated, and small enough to fine-tune briefly on one node.
# Override with EZPZ_BENCH_HF_MODEL to benchmark a different checkpoint.
HF_MODEL=${EZPZ_BENCH_HF_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
HF_DATASET=${EZPZ_BENCH_HF_DATASET:-eliplutchok/fineweb-small-sample}
HF_MAX_STEPS=${EZPZ_BENCH_HF_MAX_STEPS:-20}
HF_BLOCK_SIZE=${EZPZ_BENCH_HF_BLOCK_SIZE:-1024}

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

case "${EXAMPLE}" in
hf | hf_trainer)
    echo "ezpz ${EXAMPLE}: model=${HF_MODEL} dataset=${HF_DATASET}" \
        "max_steps=${HF_MAX_STEPS} outdir=${OUTDIR}"
    run_hf_example "${EXAMPLE}"
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
