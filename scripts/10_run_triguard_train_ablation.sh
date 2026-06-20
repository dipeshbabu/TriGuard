#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-outputs/icml2026_triguard_train_ablation_mainconf}"
DATASET="${DATASET:-cifar10}"
MODEL="${MODEL:-resnet50_imagenet}"
SEEDS="${SEEDS:-0,1,2}"
BATCH="${BATCH:-32}"
K_ATTR="${K_ATTR:-100}"
EVAL_BATCHES_ADV="${EVAL_BATCHES_ADV:-10}"
IG_STEPS="${IG_STEPS:-32}"
TRIGUARD_IG_STEPS="${TRIGUARD_IG_STEPS:-8}"
BASELINE_MODES="${BASELINE_MODES:-zero,blur,noise,uniform,mean}"
FAR_SAMPLES="${FAR_SAMPLES:-2}"

run_setting() {
  local name="$1"
  local lambda_entropy="$2"
  local lambda_wads="$3"
  local lambda_rar="$4"
  local lambda_far="$5"
  local lambda_curvature="$6"
  local lambda_robust="$7"

  echo "[TriGuard-Train ablation] $name"
  python run_triguard.py \
    --mode main \
    --out "$OUT_DIR" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --seeds "$SEEDS" \
    --lambda_entropy "$lambda_entropy" \
    --lambda_wads "$lambda_wads" \
    --lambda_rar "$lambda_rar" \
    --lambda_far "$lambda_far" \
    --lambda_curvature "$lambda_curvature" \
    --lambda_robust "$lambda_robust" \
    --triguard_ig_steps "$TRIGUARD_IG_STEPS" \
    --baseline_modes "$BASELINE_MODES" \
    --far_samples "$FAR_SAMPLES" \
    --batch "$BATCH" \
    --K_attr "$K_ATTR" \
    --ig_steps "$IG_STEPS" \
    --eval_batches_adv "$EVAL_BATCHES_ADV" \
    --skip_crown \
    --save_ckpt
}

run_setting "ce_only" 0.0 0.0 0.0 0.0 0.0 0.0
run_setting "entropy_only" 0.05 0.0 0.0 0.0 0.0 0.0
run_setting "rar_like" 0.0 0.0 0.05 0.0 0.0 0.0
run_setting "far_like" 0.0 0.0 0.0 0.05 0.0 0.0
run_setting "wads_small" 0.0 0.01 0.0 0.0 0.0 0.0
run_setting "wads_only" 0.0 0.05 0.0 0.0 0.0 0.0
run_setting "wads_curvature" 0.0 0.05 0.0 0.0 0.01 0.0
run_setting "wads_curvature_robust" 0.0 0.05 0.0 0.0 0.01 0.25

python -m triguard.stats --out "$OUT_DIR"
python -m triguard.make_figures --out "$OUT_DIR" --mode aggregate
python -m triguard.make_tables --out "$OUT_DIR" --label_prefix "$(basename "$OUT_DIR" | tr -c '[:alnum:]' '_')"
