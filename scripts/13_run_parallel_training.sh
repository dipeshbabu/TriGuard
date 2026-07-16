#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

WORKFLOW="${WORKFLOW:-primary}"
GPU_IDS="${GPU_IDS:-0}"
MAKE_ARTIFACTS_AFTER="${MAKE_ARTIFACTS_AFTER:-1}"

case "$WORKFLOW" in
  primary)
    TARGET_SCRIPT="scripts/09_run_triguard_train.sh"
    DEFAULT_SEEDS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"
    OUT_DIR="${OUT_DIR:-outputs/icml2026_triguard_train_mainconf}"
    ;;
  ablation)
    TARGET_SCRIPT="scripts/10_run_triguard_train_ablation.sh"
    DEFAULT_SEEDS="0,1,2,3,4,5,6,7,8,9"
    OUT_DIR="${OUT_DIR:-outputs/icml2026_triguard_train_ablation_mainconf}"
    ;;
  pretrained)
    TARGET_SCRIPT="scripts/06_run_pretrained_grid.sh"
    DEFAULT_SEEDS="0,1,2,3,4"
    OUT_DIR="${OUT_DIR:-outputs/icml2026}"
    ;;
  certification)
    TARGET_SCRIPT="scripts/07_run_certification_sweep.sh"
    DEFAULT_SEEDS="0,1,2,3,4"
    OUT_DIR="${OUT_DIR:-outputs/icml2026}"
    ;;
  *)
    echo "WORKFLOW must be primary, ablation, pretrained, or certification." >&2
    exit 2
    ;;
esac

SEEDS="${SEEDS:-$DEFAULT_SEEDS}"
IFS=',' read -r -a gpu_ids <<< "$GPU_IDS"
IFS=',' read -r -a seeds <<< "$SEEDS"

if [[ "${#gpu_ids[@]}" -eq 0 || "${#seeds[@]}" -eq 0 ]]; then
  echo "GPU_IDS and SEEDS must each contain at least one value." >&2
  exit 2
fi

declare -a shards
for index in "${!seeds[@]}"; do
  shard_index=$((index % ${#gpu_ids[@]}))
  current="${shards[$shard_index]-}"
  shards[$shard_index]="${current:+$current,}${seeds[$index]}"
done

declare -a pids
for index in "${!gpu_ids[@]}"; do
  shard="${shards[$index]-}"
  if [[ -z "$shard" ]]; then
    continue
  fi
  gpu="${gpu_ids[$index]}"
  echo "[parallel] GPU $gpu: seeds $shard"
  CUDA_VISIBLE_DEVICES="$gpu" \
    SEEDS="$shard" \
    OUT_DIR="$OUT_DIR" \
    MAKE_ARTIFACTS=0 \
    bash "$TARGET_SCRIPT" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one training worker failed." >&2
  exit "$status"
fi

if [[ "$MAKE_ARTIFACTS_AFTER" == "1" && "$WORKFLOW" =~ ^(primary|ablation)$ ]]; then
  OUT_DIRS="$OUT_DIR" bash scripts/11_make_triguard_train_artifacts.sh
fi
