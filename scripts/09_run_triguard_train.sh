#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

OUT_DIR="${OUT_DIR:-outputs/icml2026_triguard_train_mainconf}"
DATASET="${DATASET:-cifar10}"
MODEL="${MODEL:-resnet50_imagenet}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}"
BATCH="${BATCH:-32}"
K_ATTR="${K_ATTR:-100}"
EVAL_BATCHES_ADV="${EVAL_BATCHES_ADV:-10}"
IG_STEPS="${IG_STEPS:-32}"
TRIGUARD_IG_STEPS="${TRIGUARD_IG_STEPS:-8}"
BASELINE_MODES="${BASELINE_MODES:-zero,blur,noise,uniform,midpoint}"
FAR_SAMPLES="${FAR_SAMPLES:-2}"
REFERENCE_PAIR_SAMPLES="${REFERENCE_PAIR_SAMPLES:-0}"
REGULARIZER_MICROBATCH="${REGULARIZER_MICROBATCH:-1}"
CHECKPOINT_REGULARIZER_IG="${CHECKPOINT_REGULARIZER_IG:-1}"
SAMPLED_MASS_PENALTY="${SAMPLED_MASS_PENALTY:-0}"
PRELOAD_REFERENCE_BANKS="${PRELOAD_REFERENCE_BANKS:-1}"
MAKE_ARTIFACTS="${MAKE_ARTIFACTS:-1}"
REFERENCE_BANK="${REFERENCE_BANK:-reference_banks/mainconf/${DATASET}_${MODEL}_train_references.pt}"
HELDOUT_REFERENCE_BANK="${HELDOUT_REFERENCE_BANK:-reference_banks/mainconf/${DATASET}_${MODEL}_heldout_references.pt}"
RESERVATION_FILE="${RESERVATION_FILE:-reference_banks/mainconf/${DATASET}_${MODEL}_candidates.pt}"

reference_args=()
optimization_args=(
  --reference_pair_samples "$REFERENCE_PAIR_SAMPLES"
  --regularizer_microbatch "$REGULARIZER_MICROBATCH"
  --vectorized_reference_ig
)
if [[ "$CHECKPOINT_REGULARIZER_IG" == "1" ]]; then
  optimization_args+=(--checkpoint_regularizer_ig)
fi
if [[ "$SAMPLED_MASS_PENALTY" == "1" ]]; then
  optimization_args+=(--sampled_mass_penalty)
fi
if [[ "$PRELOAD_REFERENCE_BANKS" == "1" ]]; then
  optimization_args+=(--preload_reference_banks)
fi
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

run_primary() {
  local name="$1"
  local lambda_curvature="$2"
  local lambda_robust="$3"
  local lambda_attr_mass="$4"
  local reference_risk="$5"

  echo "[TriGuard primary] $name"
  python run_triguard.py \
    --mode main \
    --target_mode pred \
    --out "$OUT_DIR" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --seeds "$SEEDS" \
    --lambda_entropy 0.0 \
    --lambda_wads 0.05 \
    --lambda_rar 0.0 \
    --lambda_far 0.0 \
    --lambda_curvature "$lambda_curvature" \
    --lambda_robust "$lambda_robust" \
    --lambda_attr_mass "$lambda_attr_mass" \
    --attr_mass_floor 0.9 \
    --reference_risk "$reference_risk" \
    --reference_cvar_alpha 0.75 \
    --reference_bank_samples 4 \
    --eval_reference_bank_samples 16 \
    --triguard_ig_steps "$TRIGUARD_IG_STEPS" \
    --baseline_modes "$BASELINE_MODES" \
    --far_samples "$FAR_SAMPLES" \
    "${optimization_args[@]}" \
    --batch "$BATCH" \
    --num_workers 4 \
    --K_attr "$K_ATTR" \
    --K_faith 50 \
    --ig_steps "$IG_STEPS" \
    --main_faithfulness \
    --attack_suite autoattack \
    --autoattack_samples 1000 \
    --eval_batches_adv "$EVAL_BATCHES_ADV" \
    --cert_pixel_eps 0.0078431373 \
    --skip_crown \
    --save_ckpt \
    "${reference_args[@]}"
}

run_primary "mean_reference" 0.0 0.0 0.0 mean
run_primary "max_reference" 0.0 0.0 0.0 max
run_primary "max_reference_controls" 0.01 0.25 0.0 max
run_primary "cvar_mass_candidate" 0.01 0.25 0.01 cvar

if [[ "$MAKE_ARTIFACTS" == "1" ]]; then
  python -m triguard.stats --out "$OUT_DIR"
  python -m triguard.make_figures --out "$OUT_DIR" --mode aggregate
  python -m triguard.make_tables --out "$OUT_DIR" --label_prefix "$(basename "$OUT_DIR" | tr -c '[:alnum:]' '_')"
fi
