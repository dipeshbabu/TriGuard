#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

# Set any block to 0 to skip it when resuming a partial rerun.
RUN_PRETRAINED_GRID="${RUN_PRETRAINED_GRID:-1}"
RUN_CERT_SWEEP="${RUN_CERT_SWEEP:-1}"
RUN_RESERVATION="${RUN_RESERVATION:-1}"
RUN_CALIBRATION="${RUN_CALIBRATION:-1}"
RUN_REFERENCE_BANK="${RUN_REFERENCE_BANK:-1}"
RUN_TRIGUARD_TRAIN="${RUN_TRIGUARD_TRAIN:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"
MAKE_ARTIFACTS="${MAKE_ARTIFACTS:-1}"

REFERENCE_DIR="${REFERENCE_DIR:-reference_banks/mainconf}"
RESERVATION_FILE="${RESERVATION_FILE:-$REFERENCE_DIR/cifar10_resnet50_imagenet_candidates.pt}"
CALIBRATION_OUT="${CALIBRATION_OUT:-outputs/icml2026_calibration}"
CALIBRATION_CHECKPOINT="${CALIBRATION_CHECKPOINT:-$CALIBRATION_OUT/checkpoints/main_cifar10_resnet50_imagenet_seed0_lam0.000.pt}"
REFERENCE_BANK="${REFERENCE_BANK:-$REFERENCE_DIR/cifar10_resnet50_imagenet_train_references.pt}"
HELDOUT_REFERENCE_BANK="${HELDOUT_REFERENCE_BANK:-$REFERENCE_DIR/cifar10_resnet50_imagenet_heldout_references.pt}"

if [[ "$RUN_RESERVATION" == "1" ]]; then
  OUT_FILE="$RESERVATION_FILE" DATASET=cifar10 MODEL=resnet50_imagenet \
    bash scripts/11_reserve_reference_candidates.sh
fi

if [[ "$RUN_CALIBRATION" == "1" ]]; then
  python run_triguard.py \
    --mode main \
    --target_mode pred \
    --out "$CALIBRATION_OUT" \
    --dataset cifar10 \
    --model resnet50_imagenet \
    --seed 0 \
    --lambda_entropy 0.0 \
    --batch 32 \
    --num_workers 4 \
    --K_attr 1 \
    --eval_batches_adv 1 \
    --pgd_steps 1 \
    --pgd_restarts 1 \
    --skip_crown \
    --save_ckpt \
    --checkpoint_path "$CALIBRATION_CHECKPOINT" \
    --exclude_train_indices_file "$RESERVATION_FILE"
fi

if [[ "$RUN_PRETRAINED_GRID" == "1" ]]; then
  bash scripts/06_run_pretrained_grid.sh
fi

if [[ "$RUN_CERT_SWEEP" == "1" ]]; then
  bash scripts/07_run_certification_sweep.sh
fi

if [[ "$RUN_REFERENCE_BANK" == "1" ]]; then
  CHECKPOINT="$CALIBRATION_CHECKPOINT" OUT_DIR="$REFERENCE_DIR" \
    CANDIDATE_INDICES="$RESERVATION_FILE" DATASET=cifar10 MODEL=resnet50_imagenet \
    bash scripts/11_build_reference_banks.sh
fi

if [[ "$RUN_TRIGUARD_TRAIN" == "1" ]]; then
  REFERENCE_BANK="$REFERENCE_BANK" HELDOUT_REFERENCE_BANK="$HELDOUT_REFERENCE_BANK" RESERVATION_FILE="$RESERVATION_FILE" \
    bash scripts/09_run_triguard_train.sh
fi

if [[ "$RUN_ABLATION" == "1" ]]; then
  REFERENCE_BANK="$REFERENCE_BANK" HELDOUT_REFERENCE_BANK="$HELDOUT_REFERENCE_BANK" RESERVATION_FILE="$RESERVATION_FILE" \
    bash scripts/10_run_triguard_train_ablation.sh
fi

if [[ "$MAKE_ARTIFACTS" == "1" ]]; then
  bash scripts/11_make_triguard_train_artifacts.sh
fi
