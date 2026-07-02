#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

python run_triguard.py \
  --mode faithfulness \
  --out outputs/icml2026 \
  --seed 0 \
  --K_faith 50 \
  --save_ckpt
