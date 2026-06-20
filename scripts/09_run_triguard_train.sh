#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-outputs/icml2026_triguard_train}"
SEEDS="${SEEDS:-0,1,2,3,4}"
BATCH="${BATCH:-32}"
K_ATTR="${K_ATTR:-100}"
EVAL_BATCHES_ADV="${EVAL_BATCHES_ADV:-10}"
IG_STEPS="${IG_STEPS:-32}"
TRIGUARD_IG_STEPS="${TRIGUARD_IG_STEPS:-8}"
BASELINE_MODES="${BASELINE_MODES:-zero,blur,noise,uniform,mean}"

python run_triguard.py \
  --mode main \
  --out "$OUT_DIR" \
  --grid pretrained \
  --seeds "$SEEDS" \
  --lambda_entropy 0.0 \
  --lambda_wads 0.05 \
  --lambda_curvature 0.01 \
  --lambda_robust 0.25 \
  --triguard_ig_steps "$TRIGUARD_IG_STEPS" \
  --baseline_modes "$BASELINE_MODES" \
  --batch "$BATCH" \
  --K_attr "$K_ATTR" \
  --ig_steps "$IG_STEPS" \
  --eval_batches_adv "$EVAL_BATCHES_ADV" \
  --cert_pixel_eps 0.0078431373 \
  --skip_crown \
  --save_ckpt
