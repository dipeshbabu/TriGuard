#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-outputs/icml2026}"
LABEL_PREFIX="${LABEL_PREFIX:-$(basename "$OUT_DIR" | tr -c '[:alnum:]' '_')}"

python -m triguard.stats --out "$OUT_DIR"
python -m triguard.make_figures --out "$OUT_DIR" --mode aggregate
python -m triguard.make_tables --out "$OUT_DIR" --label_prefix "$LABEL_PREFIX"
