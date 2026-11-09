#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${MODE:-profile}"
CONFIG="${CONFIG:-configs/quality_audit.yaml}"

ARGS=("$MODE" --config "$CONFIG")

if [[ "${OVERWRITE:-0}" == "1" ]]; then
    ARGS+=(--overwrite)
fi

mkdir -p artifacts/logs/quality_audit
LOG_FILE="artifacts/logs/quality_audit/${MODE}_$(date +%Y%m%d_%H%M%S)_$$.log"

"$PYTHON_BIN" -u src/audit/quality_audit.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"