#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

OUT_DIR="${OUT_DIR:-outputs/icml2026_triguard_train_ablation_mainconf}"
DATASET="${DATASET:-cifar10}"
MODEL="${MODEL:-resnet50_imagenet}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
BATCH="${BATCH:-32}"
K_ATTR="${K_ATTR:-100}"
EVAL_BATCHES_ADV="${EVAL_BATCHES_ADV:-10}"
IG_STEPS="${IG_STEPS:-32}"
TRIGUARD_IG_STEPS="${TRIGUARD_IG_STEPS:-8}"
BASELINE_MODES="${BASELINE_MODES:-zero,blur,noise,uniform,midpoint}"
FAR_SAMPLES="${FAR_SAMPLES:-2}"
REFERENCE_BANK="${REFERENCE_BANK:-reference_banks/mainconf/${DATASET}_${MODEL}_train_references.pt}"
HELDOUT_REFERENCE_BANK="${HELDOUT_REFERENCE_BANK:-reference_banks/mainconf/${DATASET}_${MODEL}_heldout_references.pt}"
RESERVATION_FILE="${RESERVATION_FILE:-reference_banks/mainconf/${DATASET}_${MODEL}_candidates.pt}"

reference_args=()
if [[ -n "$REFERENCE_BANK" ]]; then
  reference_args+=(--reference_bank "$REFERENCE_BANK")
  if [[ ",$BASELINE_MODES," != *,bank,* ]]; then
    BASELINE_MODES="$BASELINE_MODES,bank"
  fi
fi
if [[ -n "$HELDOUT_REFERENCE_BANK" ]]; then
  reference_args+=(--heldout_reference_bank "$HELDOUT_REFERENCE_BANK")
fi
if [[ -n "$RESERVATION_FILE" ]]; then
  reference_args+=(--exclude_train_indices_file "$RESERVATION_FILE")
fi

run_setting() {
  local name="$1"
  local lambda_entropy="$2"
  local lambda_wads="$3"
  local lambda_rar="$4"
  local lambda_far="$5"
  local lambda_curvature="$6"
  local lambda_robust="$7"
  local lambda_attr_mass="$8"
  local reference_risk="$9"

  echo "[TriGuard-Train ablation] $name"
  python run_triguard.py \
    --mode main \
    --target_mode pred \
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
    --lambda_attr_mass "$lambda_attr_mass" \
    --reference_risk "$reference_risk" \
    --reference_cvar_alpha 0.75 \
    --reference_bank_samples 4 \
    --eval_reference_bank_samples 16 \
    --triguard_ig_steps "$TRIGUARD_IG_STEPS" \
    --baseline_modes "$BASELINE_MODES" \
    --far_samples "$FAR_SAMPLES" \
    --batch "$BATCH" \
    --num_workers 4 \
    --K_attr "$K_ATTR" \
    --ig_steps "$IG_STEPS" \
    --attack_suite autoattack \
    --autoattack_samples 1000 \
    --eval_batches_adv "$EVAL_BATCHES_ADV" \
    --skip_crown \
    --save_ckpt \
    "${reference_args[@]}"
}

run_setting "ce_only" 0.0 0.0 0.0 0.0 0.0 0.0 0.0 max
run_setting "entropy_only" 0.05 0.0 0.0 0.0 0.0 0.0 0.0 max
run_setting "rar_like" 0.0 0.0 0.05 0.0 0.0 0.0 0.0 max
run_setting "far_like" 0.0 0.0 0.0 0.05 0.0 0.0 0.0 max
run_setting "wads_small" 0.0 0.01 0.0 0.0 0.0 0.0 0.0 max
run_setting "mean_reference" 0.0 0.05 0.0 0.0 0.0 0.0 0.0 mean
run_setting "wads_only" 0.0 0.05 0.0 0.0 0.0 0.0 0.0 max
run_setting "wads_curvature" 0.0 0.05 0.0 0.0 0.01 0.0 0.0 max
run_setting "wads_curvature_robust" 0.0 0.05 0.0 0.0 0.01 0.25 0.0 max
run_setting "cvar_curvature_robust" 0.0 0.05 0.0 0.0 0.01 0.25 0.0 cvar
run_setting "cvar_mass_candidate" 0.0 0.05 0.0 0.0 0.01 0.25 0.01 cvar

python -m triguard.stats --out "$OUT_DIR"
python -m triguard.make_figures --out "$OUT_DIR" --mode aggregate
python -m triguard.make_tables --out "$OUT_DIR" --label_prefix "$(basename "$OUT_DIR" | tr -c '[:alnum:]' '_')"
