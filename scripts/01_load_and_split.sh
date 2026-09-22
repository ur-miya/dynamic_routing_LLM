#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-configs/data.yaml}"
LOG_DIR="${LOG_DIR:-artifacts/logs/data}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="${LOG_DIR}/01_load_and_split_${RUN_ID}.log"
mkdir -p "${LOG_DIR}" data/raw data/interim data/processed artifacts/data
exec > >(tee -a "${RUN_LOG}") 2>&1

on_error() {
  local exit_code=$?
  echo "[$(date -u +%FT%TZ)] ERROR line=${BASH_LINENO[0]} exit_code=${exit_code}"
  exit "${exit_code}"
}
trap on_error ERR

echo "[$(date -u +%FT%TZ)] Starting OASST1 data stage"
echo "Project root: ${PROJECT_ROOT}"
echo "Config: ${CONFIG}"
echo "Python: $(${PYTHON_BIN} --version 2>&1)"

"${PYTHON_BIN}" - <<'PY'
import importlib
for package in ("datasets", "yaml", "numpy", "sklearn"):
    importlib.import_module(package)
print("Dependency check passed")
PY

COMMON_ARGS=(--config "${CONFIG}" --log-level "${LOG_LEVEL:-INFO}")
OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi
LOAD_EXTRA_ARGS=()
if [[ -n "${MAX_MESSAGES:-}" ]]; then
  LOAD_EXTRA_ARGS+=(--max-messages "${MAX_MESSAGES}")
fi
if [[ -n "${DATASET_REVISION:-}" ]]; then
  LOAD_EXTRA_ARGS+=(--revision "${DATASET_REVISION}")
fi

"${PYTHON_BIN}" src/data/load_oasst1.py \
  "${COMMON_ARGS[@]}" "${OVERWRITE_ARGS[@]}" "${LOAD_EXTRA_ARGS[@]}"

"${PYTHON_BIN}" src/data/build_pairs.py \
  "${COMMON_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

"${PYTHON_BIN}" src/data/split.py \
  "${COMMON_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

echo "[$(date -u +%FT%TZ)] OASST1 data stage completed successfully"
echo "Run log: ${RUN_LOG}"