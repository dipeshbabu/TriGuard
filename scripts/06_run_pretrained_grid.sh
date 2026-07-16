#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

OUT_DIR="${OUT_DIR:-outputs/icml2026}"
SEEDS="${SEEDS:-0,1,2,3,4}"

python run_triguard.py \
  --mode main \
  --target_mode pred \
  --out "$OUT_DIR" \
  --grid pretrained \
  --seeds "$SEEDS" \
  --lambda_entropy 0.0 \
  --batch 32 \
  --K_attr 100 \
  --eval_batches_adv 10 \
  --cert_pixel_eps 0.0078431373 \
  --skip_crown \
  --save_ckpt
