#!/usr/bin/env bash
# Fetch neural repos + checkpoints for Spectroformer / NU2Net looks.
# Safe to re-run; skips work that is already present.
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_CLONE=0
if [[ "${1:-}" == "--skip-clone" ]]; then
  SKIP_CLONE=1
fi

spectro_repo="eval/repos/spectroformer"
nu2_repo="eval/repos/uie_benchmark"
spectro_best="$spectro_repo/checkpoints/best.pth"
nu2_ckpt="$nu2_repo/checkpoints/UIEB/NU2Net.ckpt"

clone_if_missing() {
  local url="$1" dest="$2" name="$3"
  if [[ -d "$dest/.git" ]]; then
    echo "[ok] $name already cloned: $dest"
    return
  fi
  if [[ -e "$dest" ]]; then
    echo "[warn] $dest exists but is not a git clone; leave as-is"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  echo "[fetch] cloning $name ..."
  git clone --depth 1 "$url" "$dest"
}

if [[ $SKIP_CLONE -eq 0 ]]; then
  command -v git >/dev/null || { echo "git is required on PATH"; exit 1; }
  clone_if_missing 'https://github.com/Mdraqibkhan/Spectroformer.git' "$spectro_repo" 'Spectroformer'
  clone_if_missing 'https://github.com/ddz16/Underwater-Image-Enhancement-Benchmark.git' "$nu2_repo" 'uie_benchmark (NU2Net)'
fi

mkdir -p "$spectro_repo/checkpoints"
if [[ -f "$spectro_best" ]]; then
  echo "[ok] Spectroformer weights: $spectro_best"
else
  pick="$(find "$spectro_repo/checkpoints" -type f -name '*.pth' ! -name 'best.pth' 2>/dev/null | head -n 1 || true)"
  if [[ -n "$pick" ]]; then
    cp "$pick" "$spectro_best"
    echo "[ok] copied $pick -> checkpoints/best.pth"
  else
    echo "[missing] $spectro_best"
    echo "         Place the validated UIEB (or bakeoff) checkpoint at that path."
    echo "         See docs/neural-setup.md and THIRD_PARTY.md."
  fi
fi

if [[ -f "$nu2_ckpt" ]]; then
  echo "[ok] NU2Net weights: $nu2_ckpt"
else
  echo "[missing] $nu2_ckpt"
  echo "         Download NU2Net.ckpt into eval/repos/uie_benchmark/checkpoints/UIEB/"
fi

echo ""
echo "Next: bash scripts/setup.sh"
echo "Then: bash scripts/process.sh /path/to/NEFs"
