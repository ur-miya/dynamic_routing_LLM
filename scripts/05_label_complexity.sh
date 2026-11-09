#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${MODE:-inspect}"
CONFIG="${CONFIG:-configs/labeling.yaml}"

args=(--config "$CONFIG")

if [[ -n "${SPLITS:-}" ]]; then
    read -r -a split_names <<< "$SPLITS"
    args+=(--splits "${split_names[@]}")
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
    args+=(--smoke --limit "${LIMIT:-8}")
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
    args+=(--overwrite)
fi

case "$MODE" in
    inspect) args+=(--inspect) ;;
    smoke)
        if [[ "${SMOKE:-0}" != "1" ]]; then
            args+=(--smoke --limit "${LIMIT:-8}")
        fi
        ;;
    label) ;;
    validate) args+=(--validate) ;;
    *)
        echo "Unknown MODE=$MODE; use inspect|smoke|label|validate" >&2
        exit 2
        ;;
esac

mkdir -p artifacts/logs/labeling
LOG_FILE="artifacts/logs/labeling/${MODE}_$(date +%Y%m%d_%H%M%S)_$$.log"

"$PYTHON_BIN" -u src/labeling/label_complexity.py "${args[@]}" 2>&1 | tee "$LOG_FILE"