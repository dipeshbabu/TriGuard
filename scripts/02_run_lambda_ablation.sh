#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

python run_triguard.py \
  --mode lambda \
  --target_mode pred \
  --out outputs/icml2026 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --epochs 5 \
  --dataset mnist \
  --model simplecnn \
  --lambda_list "0.0,0.01,0.05,0.1" \
  --K_attr 100 \
  --save_ckpt
