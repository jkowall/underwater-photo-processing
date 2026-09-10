#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON_BIN:-python3.12}"
"$python_bin" -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12 or newer is required"'
"$python_bin" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
printf 'Ready. Run .venv/bin/python scripts/process_batch.py --help\n'
