#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_VERSION="$(tr -d '[:space:]' < "$REPO_ROOT/.python-version")"

if [[ -x "$REPO_ROOT/.triguard/bin/python" ]]; then
  source "$REPO_ROOT/.triguard/bin/activate"
elif [[ -x "$REPO_ROOT/.triguard/Scripts/python.exe" ]]; then
  source "$REPO_ROOT/.triguard/Scripts/activate"
else
  echo "Missing .triguard environment. Run: bash scripts/00_install.sh" >&2
  exit 1
fi

ACTIVE_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ "$ACTIVE_VERSION" != "$PYTHON_VERSION" ]]; then
  echo "TriGuard requires Python $PYTHON_VERSION, but .triguard has Python $ACTIVE_VERSION." >&2
  echo "Recreate it with: bash scripts/00_install.sh" >&2
  exit 1
fi
