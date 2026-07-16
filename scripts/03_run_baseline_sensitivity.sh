#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

python run_triguard.py \
  --mode baseline \
  --target_mode pred \
  --out outputs/icml2026 \
  --seed 0 \
  --dataset cifar10 \
  --model resnet50 \
  --K_attr 100 \
  --save_ckpt
