python run_triguard.py \
  --mode lambda \
  --out outputs/icml2026 \
  --seeds 0,1,2 \
  --epochs 5 \
  --dataset mnist \
  --model simplecnn \
  --lambda_list "0.0,0.01,0.05,0.1" \
  --K_attr 100 \
  --save_ckpt
