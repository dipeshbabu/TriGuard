source .venv/bin/activate

# Baseline sensitivity: recommend MNIST + SimpleCNN (writes outputs/icml2026/table2_baseline_sensitivity.csv)
python -m triguard.run_all --mode baseline --out outputs/icml2026 --seed 0 --epochs 5 --dataset mnist --model simplecnn --K_attr 100
