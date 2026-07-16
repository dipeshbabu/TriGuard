#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="$(tr -d '[:space:]' < .python-version)"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

uv python install "$PYTHON_VERSION" --managed-python
uv venv --python "$PYTHON_VERSION" --managed-python --allow-existing .triguard

if [[ -x ".triguard/bin/python" ]]; then
  VENV_PYTHON=".triguard/bin/python"
  VENV_ACTIVATE=".triguard/bin/activate"
elif [[ -x ".triguard/Scripts/python.exe" ]]; then
  VENV_PYTHON=".triguard/Scripts/python.exe"
  VENV_ACTIVATE=".triguard/Scripts/activate"
else
  echo "Could not find the Python executable in .triguard" >&2
  return 1 2>/dev/null || exit 1
fi

ACTIVE_VERSION="$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ "$ACTIVE_VERSION" != "$PYTHON_VERSION" ]]; then
  echo "Recreating .triguard with Python $PYTHON_VERSION (found $ACTIVE_VERSION)." >&2
  uv venv --python "$PYTHON_VERSION" --managed-python --clear .triguard
fi

source "$VENV_ACTIVATE"

uv pip install --python "$VENV_PYTHON" setuptools wheel

uv pip install --python "$VENV_PYTHON" \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url "$TORCH_INDEX_URL"

uv pip install --python "$VENV_PYTHON" -r requirements.txt

uv pip install --python "$VENV_PYTHON" ./auto_LiRPA
uv pip install --python "$VENV_PYTHON" -r requirements-autoattack.txt
uv pip install --python "$VENV_PYTHON" -r requirements-dev.txt
uv pip check --python "$VENV_PYTHON"

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Setup complete. Activated: $VIRTUAL_ENV"
elif [[ -t 0 && -t 1 ]]; then
  echo "Setup complete. Starting an activated shell. Run 'exit' to return to your previous shell."
  exec "${BASH:-bash}" --init-file "$VENV_ACTIVATE" -i
else
  echo "Setup complete. Activate the environment with: source $VENV_ACTIVATE"
fi
