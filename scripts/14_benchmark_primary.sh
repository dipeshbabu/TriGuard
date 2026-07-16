#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

OUT_DIR="${OUT_DIR:-outputs/icml2026_optimization_benchmark}"
DATASET="${DATASET:-cifar10}"
MODEL="${MODEL:-resnet50_imagenet}"
BATCH="${BATCH:-32}"
REGULARIZER_MICROBATCH="${REGULARIZER_MICROBATCH:-1}"
REFERENCE_PAIR_SAMPLES="${REFERENCE_PAIR_SAMPLES:-0}"
SAMPLED_MASS_PENALTY="${SAMPLED_MASS_PENALTY:-0}"
GPU_COUNT="${GPU_COUNT:-8}"
RUN_FULL_EVAL_BENCHMARK="${RUN_FULL_EVAL_BENCHMARK:-1}"
EPOCHS_PER_RUN="${EPOCHS_PER_RUN:-20}"
PRIMARY_RUNS="${PRIMARY_RUNS:-80}"
WADS_RUNS="${WADS_RUNS:-150}"
REFERENCE_BANK="${REFERENCE_BANK:-reference_banks/mainconf/${DATASET}_${MODEL}_train_references.pt}"
HELDOUT_REFERENCE_BANK="${HELDOUT_REFERENCE_BANK:-reference_banks/mainconf/${DATASET}_${MODEL}_heldout_references.pt}"
RESERVATION_FILE="${RESERVATION_FILE:-reference_banks/mainconf/${DATASET}_${MODEL}_candidates.pt}"
BENCHMARK_CHECKPOINT="${BENCHMARK_CHECKPOINT:-$OUT_DIR/benchmark_checkpoint.pt}"

sampling_args=()
if [[ "$SAMPLED_MASS_PENALTY" == "1" ]]; then
  sampling_args+=(--sampled_mass_penalty)
fi

python run_triguard.py \
  --mode main \
  --out "$OUT_DIR" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --seed 0 \
  --epochs 1 \
  --batch "$BATCH" \
  --num_workers 4 \
  --lambda_entropy 0.0 \
  --lambda_wads 0.05 \
  --lambda_curvature 0.01 \
  --lambda_robust 0.25 \
  --lambda_attr_mass 0.01 \
  --reference_risk cvar \
  --reference_cvar_alpha 0.75 \
  --reference_pair_samples "$REFERENCE_PAIR_SAMPLES" \
  "${sampling_args[@]}" \
  --reference_bank_samples 4 \
  --regularizer_microbatch "$REGULARIZER_MICROBATCH" \
  --vectorized_reference_ig \
  --checkpoint_regularizer_ig \
  --preload_reference_banks \
  --triguard_ig_steps 8 \
  --baseline_modes zero,blur,noise,uniform,midpoint,bank \
  --reference_bank "$REFERENCE_BANK" \
  --heldout_reference_bank "$HELDOUT_REFERENCE_BANK" \
  --exclude_train_indices_file "$RESERVATION_FILE" \
  --attack_suite pgd \
  --pgd_steps 1 \
  --pgd_restarts 1 \
  --eval_batches_adv 1 \
  --K_attr 1 \
  --ig_steps 4 \
  --skip_crown \
  --save_ckpt \
  --checkpoint_path "$BENCHMARK_CHECKPOINT"

FULL_EVAL_OUT="$OUT_DIR/full_evaluation"
if [[ "$RUN_FULL_EVAL_BENCHMARK" == "1" ]]; then
  python run_triguard.py \
    --mode main \
    --out "$FULL_EVAL_OUT" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --seed 0 \
    --eval_only \
    --load_ckpt "$BENCHMARK_CHECKPOINT" \
    --lambda_entropy 0.0 \
    --lambda_wads 0.05 \
    --lambda_curvature 0.01 \
    --lambda_robust 0.25 \
    --lambda_attr_mass 0.01 \
    --reference_risk cvar \
    --reference_cvar_alpha 0.75 \
    --reference_pair_samples "$REFERENCE_PAIR_SAMPLES" \
    "${sampling_args[@]}" \
    --reference_bank_samples 4 \
    --eval_reference_bank_samples 16 \
    --regularizer_microbatch "$REGULARIZER_MICROBATCH" \
    --vectorized_reference_ig \
    --checkpoint_regularizer_ig \
    --preload_reference_banks \
    --triguard_ig_steps 8 \
    --baseline_modes zero,blur,noise,uniform,midpoint,bank \
    --reference_bank "$REFERENCE_BANK" \
    --heldout_reference_bank "$HELDOUT_REFERENCE_BANK" \
    --exclude_train_indices_file "$RESERVATION_FILE" \
    --attack_suite autoattack \
    --autoattack_samples 1000 \
    --autoattack_batch 128 \
    --K_attr 100 \
    --K_faith 50 \
    --ig_steps 32 \
    --delins_steps 50 \
    --main_faithfulness \
    --skip_crown
fi

BENCHMARK_OUT="$OUT_DIR" FULL_EVAL_OUT="$FULL_EVAL_OUT" \
RUN_FULL_EVAL_BENCHMARK="$RUN_FULL_EVAL_BENCHMARK" GPU_COUNT="$GPU_COUNT" \
EPOCHS_PER_RUN="$EPOCHS_PER_RUN" PRIMARY_RUNS="$PRIMARY_RUNS" \
WADS_RUNS="$WADS_RUNS" python - <<'PY'
import os

import pandas as pd

train_row = pd.read_csv(
    os.path.join(os.environ["BENCHMARK_OUT"], "table1_main.csv")
).iloc[-1]
epoch_seconds = float(train_row["train_seconds"])
eval_seconds = 0.0
peak_cuda_memory_mb = float(train_row["peak_cuda_memory_mb"])
if os.environ["RUN_FULL_EVAL_BENCHMARK"] == "1":
    eval_row = pd.read_csv(
        os.path.join(os.environ["FULL_EVAL_OUT"], "table1_main.csv")
    ).iloc[-1]
    eval_seconds = float(eval_row["eval_seconds"])
    peak_cuda_memory_mb = max(
        peak_cuda_memory_mb,
        float(eval_row["peak_cuda_memory_mb"]),
    )
epochs_per_run = int(os.environ["EPOCHS_PER_RUN"])
primary_runs = int(os.environ["PRIMARY_RUNS"])
wads_runs = int(os.environ["WADS_RUNS"])
per_run_seconds = epoch_seconds * epochs_per_run + eval_seconds
primary_gpu_hours = per_run_seconds * primary_runs / 3600
wads_gpu_hours = per_run_seconds * wads_runs / 3600
gpu_count = max(int(os.environ["GPU_COUNT"]), 1)
print(
    "Benchmark summary:",
    f"train_seconds_per_epoch={epoch_seconds:.1f}",
    f"full_eval_seconds={eval_seconds:.1f}",
    f"peak_cuda_memory_mb={peak_cuda_memory_mb:.1f}",
)
print(
    "Projection using this expensive setting for every run:",
    f"primary_gpu_hours={primary_gpu_hours:.1f}",
    f"all_wads_gpu_hours={wads_gpu_hours:.1f}",
    f"primary_days_on_{gpu_count}_gpus={primary_gpu_hours / gpu_count / 24:.1f}",
)
if os.environ["RUN_FULL_EVAL_BENCHMARK"] != "1":
    print("Warning: full evaluation was skipped, so projected time excludes evaluation.")
PY
