#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

python run_triguard.py \
  --mode main \
  --grid pretrained \
  --seeds 0,1,2,3,4 \
  --lambda_entropy 0.0 \
  --batch 32 \
  --K_attr 100 \
  --eval_batches_adv 10 \
  --cert_pixel_eps 0.0078431373 \
  --skip_crown \
  --save_ckpt
