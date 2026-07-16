#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

OUT_DIR="${OUT_DIR:-outputs/icml2026}"
SEEDS="${SEEDS:-0,1,2,3,4}"

python run_triguard.py \
  --mode cert_sweep \
  --out "$OUT_DIR" \
  --dataset mnist \
  --model simplecnn \
  --seeds "$SEEDS" \
  --lambda_entropy 0.0 \
  --batch 64 \
  --K_attr 100 \
  --cert_eps_list "0.0,0.01,0.03,0.1,0.3" \
  --save_ckpt
