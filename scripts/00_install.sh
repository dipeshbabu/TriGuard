#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="$(tr -d '[:space:]' < .python-version)"

uv python install "$PYTHON_VERSION" --managed-python
uv venv --python "$PYTHON_VERSION" --managed-python .triguard

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

source "$VENV_ACTIVATE"

uv pip install --python "$VENV_PYTHON" \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install --python "$VENV_PYTHON" -r requirements.txt

uv pip install --python "$VENV_PYTHON" ./auto_LiRPA

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Setup complete. Activated: $VIRTUAL_ENV"
elif [[ -t 0 && -t 1 ]]; then
  echo "Setup complete. Starting an activated shell. Run 'exit' to return to your previous shell."
  exec "${BASH:-bash}" --init-file "$VENV_ACTIVATE" -i
else
  echo "Setup complete. Activate the environment with: source $VENV_ACTIVATE"
fi
