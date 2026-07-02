#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_activate_env.sh"

OUT_DIRS="${OUT_DIRS:-outputs/icml2026 outputs/icml2026_triguard_train_mainconf outputs/icml2026_triguard_train_ablation_mainconf}"

for out_dir in $OUT_DIRS; do
  if [[ ! -d "$out_dir" ]]; then
    echo "[skip] $out_dir does not exist"
    continue
  fi

  echo "[artifacts] $out_dir"
  label_prefix="$(basename "$out_dir" | tr -c '[:alnum:]' '_')"
  python -m triguard.stats --out "$out_dir"
  python -m triguard.make_figures --out "$out_dir" --mode aggregate
  python -m triguard.make_tables --out "$out_dir" --label_prefix "$label_prefix"
done
