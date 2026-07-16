#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

DATASET="${DATASET:-cifar10}"
MODEL="${MODEL:-resnet50_imagenet}"
OUT_FILE="${OUT_FILE:-reference_banks/mainconf/${DATASET}_${MODEL}_candidates.pt}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-2048}"
SEED="${SEED:-0}"

python -m triguard.reserve_references \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --out "$OUT_FILE" \
  --candidate_count "$CANDIDATE_COUNT" \
  --seed "$SEED"
