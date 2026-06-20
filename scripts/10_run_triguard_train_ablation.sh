#!/usr/bin/env bash
set -euo pipefail

python run_triguard.py \
  --mode main \
  --out outputs/icml2026_triguard_train_ablation \
  --dataset cifar10 \
  --model resnet50_imagenet \
  --seeds 0,1,2 \
  --lambda_entropy 0.0 \
  --lambda_wads 0.01 \
  --lambda_curvature 0.0 \
  --lambda_robust 0.0 \
  --triguard_ig_steps 8 \
  --baseline_modes "zero,blur,noise,uniform,mean" \
  --batch 32 \
  --K_attr 100 \
  --eval_batches_adv 10 \
  --skip_crown \
  --save_ckpt

python run_triguard.py \
  --mode main \
  --out outputs/icml2026_triguard_train_ablation \
  --dataset cifar10 \
  --model resnet50_imagenet \
  --seeds 0,1,2 \
  --lambda_entropy 0.0 \
  --lambda_wads 0.05 \
  --lambda_curvature 0.01 \
  --lambda_robust 0.0 \
  --triguard_ig_steps 8 \
  --baseline_modes "zero,blur,noise,uniform,mean" \
  --batch 32 \
  --K_attr 100 \
  --eval_batches_adv 10 \
  --skip_crown \
  --save_ckpt

python run_triguard.py \
  --mode main \
  --out outputs/icml2026_triguard_train_ablation \
  --dataset cifar10 \
  --model resnet50_imagenet \
  --seeds 0,1,2 \
  --lambda_entropy 0.0 \
  --lambda_wads 0.05 \
  --lambda_curvature 0.01 \
  --lambda_robust 0.25 \
  --triguard_ig_steps 8 \
  --baseline_modes "zero,blur,noise,uniform,mean" \
  --batch 32 \
  --K_attr 100 \
  --eval_batches_adv 10 \
  --skip_crown \
  --save_ckpt
