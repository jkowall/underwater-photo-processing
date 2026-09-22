#!/usr/bin/env bash
# Process a NEF (or folder) with the product CLI. Default look: auto.
# Opens the output folder on success (Finder on macOS, xdg-open elsewhere).
# Pass --no-open to skip.
set -euo pipefail
cd "$(dirname "$0")/.."

LOOK=auto
OUTPUT=""
NO_OPEN=0
INPUT=""

usage() {
  cat <<'EOF'
Usage: scripts/process.sh <input.NEF|folder> [--look LOOK] [--output DIR] [--no-open]
Looks: auto (default), spectroformer, nu2net, vivid, pop, natural
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --look) LOOK="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --no-open) NO_OPEN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$INPUT" ]]; then
        echo "Unexpected argument: $1" >&2
        exit 2
      fi
      INPUT="$1"
      shift
      ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
  bash scripts/setup.sh
fi

INPUT_RESOLVED="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"
if [[ -n "$OUTPUT" ]]; then
  OUT_DIR="$OUTPUT"
else
  if [[ -d "$INPUT_RESOLVED" ]]; then
    parent="$(dirname "$INPUT_RESOLVED")"
    base="$(basename "$INPUT_RESOLVED")"
  else
    parent="$(dirname "$INPUT_RESOLVED")"
    base="$(basename "$INPUT_RESOLVED")"
    base="${base%.*}"
  fi
  OUT_DIR="$parent/${base}-${LOOK}"
fi

args=(scripts/process_batch.py --input "$INPUT" --look "$LOOK" --resume)
if [[ -n "$OUTPUT" ]]; then
  args+=(--output "$OUTPUT")
fi

.venv/bin/python "${args[@]}"
code=$?

if [[ $code -eq 0 && $NO_OPEN -eq 0 && -d "$OUT_DIR" ]]; then
  echo "Opening output folder: $OUT_DIR"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    open "$OUT_DIR"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$OUT_DIR" >/dev/null 2>&1 || true
  fi
fi

exit "$code"
