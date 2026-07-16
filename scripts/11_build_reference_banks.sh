#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

DATASET="${DATASET:-cifar10}"
MODEL="${MODEL:-resnet50_imagenet}"
CHECKPOINT="${CHECKPOINT:-}"
OUT_DIR="${OUT_DIR:-reference_banks}"
CANDIDATE_INDICES="${CANDIDATE_INDICES:-}"

: "${CHECKPOINT:?Set CHECKPOINT to a frozen target-task calibration checkpoint}"

candidate_args=()
if [[ -n "$CANDIDATE_INDICES" ]]; then
  candidate_args+=(--candidate_indices "$CANDIDATE_INDICES")
fi

python -m triguard.build_references \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT_DIR" \
  --bank_size 64 \
  --heldout_size 64 \
  --candidate_limit 2048 \
  "${candidate_args[@]}"
