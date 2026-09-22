#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON_BIN:-python3.12}"
"$python_bin" -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12 or newer is required"'
"$python_bin" -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt

# Apple Silicon / macOS: install MPS-capable torch for neural looks.
# Linux without CUDA: skip (use CPU classical looks, or install CUDA torch separately).
uname_s="$(uname -s)"
uname_m="$(uname -m)"
if [[ "$uname_s" == "Darwin" ]]; then
  echo "[setup] installing Apple Silicon / macOS neural deps (MPS)…"
  if .venv/bin/python -m pip install -r requirements-neural-macos.txt; then
    .venv/bin/python -c "import torch; print('torch', torch.__version__, 'mps', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
  else
    echo "[warn] requirements-neural-macos.txt failed; classical looks still work"
  fi
fi

if [[ -x scripts/fetch_neural_weights.sh ]]; then
  bash scripts/fetch_neural_weights.sh || true
elif [[ -f scripts/fetch_neural_weights.sh ]]; then
  bash scripts/fetch_neural_weights.sh || true
fi

spectro="eval/repos/spectroformer/checkpoints/best.pth"
echo ""
echo "Neural checklist:"
if [[ -f "$spectro" ]]; then
  echo "  [ok] Spectroformer weights: $spectro"
else
  echo "  [missing] $spectro — place bakeoff checkpoint; see docs/neural-setup.md"
fi
if [[ "$uname_s" == "Darwin" ]]; then
  echo "  Apple Silicon: neural looks use MPS when torch.backends.mps.is_available()"
  echo "  Default: bash scripts/process.sh /path/to/NEFs"
  echo "  UW-only GPU: bash scripts/process.sh /path/to/NEFs --look spectroformer"
else
  echo "  Classical ready. Neural GPU on Linux needs a CUDA torch install."
fi
printf 'Ready. Run .venv/bin/python scripts/process_batch.py --help\n'
