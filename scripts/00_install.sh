#!/usr/bin/env bash
set -euo pipefail

uv venv --python 3.10.12 .triguard

if [[ -x ".triguard/bin/python" ]]; then
  VENV_PYTHON=".triguard/bin/python"
elif [[ -x ".triguard/Scripts/python.exe" ]]; then
  VENV_PYTHON=".triguard/Scripts/python.exe"
else
  echo "Could not find the Python executable in .triguard" >&2
  exit 1
fi

uv pip install --python "$VENV_PYTHON" \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install --python "$VENV_PYTHON" -r requirements.txt

uv pip install --python "$VENV_PYTHON" ./auto_LiRPA
