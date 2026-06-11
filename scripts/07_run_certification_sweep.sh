python run_triguard.py \
  --mode cert_sweep \
  --dataset mnist \
  --model simplecnn \
  --seeds 0,1,2,3,4 \
  --lambda_entropy 0.0 \
  --batch 64 \
  --K_attr 100 \
  --cert_eps_list "0.0,0.01,0.03,0.1,0.3" \
  --save_ckpt
