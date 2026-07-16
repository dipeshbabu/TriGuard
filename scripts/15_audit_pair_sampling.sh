#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

CHECKPOINT="${CHECKPOINT:-}"
DATASET="${DATASET:-cifar10}"
MODEL="${MODEL:-resnet50_imagenet}"
REFERENCE_BANK="${REFERENCE_BANK:-reference_banks/mainconf/${DATASET}_${MODEL}_train_references.pt}"
OUT_DIR="${OUT_DIR:-outputs/pair_sampling_audit}"
EXAMPLES="${EXAMPLES:-20}"
TRIALS="${TRIALS:-1000}"
IG_STEPS="${IG_STEPS:-8}"
SAMPLE_SIZES="${SAMPLE_SIZES:-4,8}"
FAIL_ON_THRESHOLD="${FAIL_ON_THRESHOLD:-0}"

: "${CHECKPOINT:?Set CHECKPOINT to a frozen TriGuard checkpoint}"

threshold_args=()
if [[ "$FAIL_ON_THRESHOLD" == "1" ]]; then
  threshold_args+=(--fail_on_threshold)
fi

python -m triguard.audit_pair_sampling \
  --checkpoint "$CHECKPOINT" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --reference_bank "$REFERENCE_BANK" \
  --out "$OUT_DIR" \
  --examples "$EXAMPLES" \
  --trials "$TRIALS" \
  --ig_steps "$IG_STEPS" \
  --sample_sizes "$SAMPLE_SIZES" \
  "${threshold_args[@]}"
