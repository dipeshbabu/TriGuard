#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

python run_triguard.py \
  --mode main \
  --target_mode pred \
  --out outputs/icml2026 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --lambda_entropy 0.05 \
  --K_attr 100 \
  --save_ckpt
