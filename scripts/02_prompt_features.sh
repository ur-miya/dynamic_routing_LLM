#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"; CONFIG="${CONFIG:-configs/features.yaml}"; MODE="${MODE:-core}"
COMMON=(--config "$CONFIG" --log-level "${LOG_LEVEL:-INFO}")
[[ -n "${SPLITS:-}" ]] && read -r -a S <<< "$SPLITS" && COMMON+=(--splits "${S[@]}")
[[ -n "${LIMIT:-}" ]] && COMMON+=(--limit "$LIMIT")
[[ "${OVERWRITE:-0}" == 1 ]] && COMMON+=(--overwrite)
ASSEMBLE_OVERWRITE_ARGS=()
[[ "${OVERWRITE:-0}" == 1 ]] && ASSEMBLE_OVERWRITE_ARGS+=(--overwrite)
case "$MODE" in
  base) "$PYTHON_BIN" src/features/prompt_features.py --base-only "${COMMON[@]}" ;;
  embeddings) "$PYTHON_BIN" src/features/prompt_features.py --embeddings-only "${COMMON[@]}" ;;
  core) "$PYTHON_BIN" src/features/prompt_features.py --base-only "${COMMON[@]}"; "$PYTHON_BIN" src/features/prompt_features.py --embeddings-only "${COMMON[@]}" ;;
  judge) "$PYTHON_BIN" src/features/prompt_features.py --judge-only --env-file "${ENV_FILE:-.env}" --preflight "${COMMON[@]}" ;;
  uncertainty) "$PYTHON_BIN" src/features/prompt_features.py --uncertainty-only "${COMMON[@]}" ;;
  assemble) "$PYTHON_BIN" src/features/assemble_prompt_features.py --config "$CONFIG" ${INCLUDE:+--include $INCLUDE} ${SPLITS:+--splits $SPLITS} ${ASSEMBLE_OVERWRITE_ARGS[@]} ;;
  *) echo "MODE must be base|embeddings|core|judge|uncertainty|assemble" >&2; exit 2 ;;
esac
