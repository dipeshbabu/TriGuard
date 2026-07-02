#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

# Set any block to 0 to skip it when resuming a partial rerun.
RUN_PRETRAINED_GRID="${RUN_PRETRAINED_GRID:-1}"
RUN_CERT_SWEEP="${RUN_CERT_SWEEP:-1}"
RUN_TRIGUARD_TRAIN="${RUN_TRIGUARD_TRAIN:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"
MAKE_ARTIFACTS="${MAKE_ARTIFACTS:-1}"

if [[ "$RUN_PRETRAINED_GRID" == "1" ]]; then
  bash scripts/06_run_pretrained_grid.sh
fi

if [[ "$RUN_CERT_SWEEP" == "1" ]]; then
  bash scripts/07_run_certification_sweep.sh
fi

if [[ "$RUN_TRIGUARD_TRAIN" == "1" ]]; then
  bash scripts/09_run_triguard_train.sh
fi

if [[ "$RUN_ABLATION" == "1" ]]; then
  bash scripts/10_run_triguard_train_ablation.sh
fi

if [[ "$MAKE_ARTIFACTS" == "1" ]]; then
  bash scripts/11_make_triguard_train_artifacts.sh
fi
