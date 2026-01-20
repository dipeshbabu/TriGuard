source .venv/bin/activate

# Main table: all datasets x models (writes outputs/icml2026/table1_main.csv)
python -m triguard.run_all --mode main --out outputs/icml2026 --seed 0 --epochs 5 --lambda_entropy 0.05 --K_attr 100
