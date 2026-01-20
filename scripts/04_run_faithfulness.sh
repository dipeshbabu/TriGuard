source .venv/bin/activate

# Faithfulness AUC + curves (writes outputs/icml2026/tables5_6_faithfulness.csv and PNG curves)
python -m triguard.run_all --mode faithfulness --out outputs/icml2026 --seed 0 --epochs 5 --K_faith 50
