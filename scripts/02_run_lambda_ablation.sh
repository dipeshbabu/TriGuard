source .venv/bin/activate

# Lambda ablation: recommend MNIST + SimpleCNN (writes outputs/icml2026/table4_lambda_ablation.csv)
python - m triguard.run_all - -mode lambda --out outputs/icml2026 - -seed 0 - -epochs 5 - -dataset mnist - -model simplecnn - -lambda_list "0.0,0.01,0.05,0.1" - -K_attr 100
