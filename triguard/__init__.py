import sys


SUPPORTED_PYTHON = (3, 10)
SUPPORTED_PYTHON_STR = "3.10.x"
TESTED_PYTHON_STR = "3.10.12"

if sys.version_info[:2] != SUPPORTED_PYTHON:
    raise RuntimeError(
        "TriGuard requires Python "
        f"{SUPPORTED_PYTHON_STR} (tested with {TESTED_PYTHON_STR}). "
        f"Current interpreter: {sys.version.split()[0]}"
    )


__version__ = "0.1.0"
