#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-configs/judge_batch_v1.yaml}"
MODE="${MODE:-smoke}"
DEPLOYMENT_TAG="${DEPLOYMENT_TAG:-weak-v1-20261101}"
mkdir -p artifacts/logs/judge_batch_v1
LOG="artifacts/logs/judge_batch_v1/${MODE}_$(date +%Y%m%d_%H%M%S)_$$.log"
args=(--config "$CONFIG")
case "$MODE" in
  prepare) "$PYTHON_BIN" -u scripts/judge_batch_v1.py prepare "${args[@]}" 2>&1 | tee "$LOG" ;;
  status) "$PYTHON_BIN" -u scripts/judge_batch_v1.py status "${args[@]}" 2>&1 | tee "$LOG" ;;
  smoke)
    "$PYTHON_BIN" -u scripts/judge_batch_v1.py prepare "${args[@]}" 2>&1 | tee "$LOG"
    "$PYTHON_BIN" -u scripts/judge_batch_v1.py run "${args[@]}" --allow-network --deployment-tag "$DEPLOYMENT_TAG" --splits train --max-new-tasks 8 --max-http-attempts 16 2>&1 | tee -a "$LOG"
    ;;
  full)
    "$PYTHON_BIN" -u scripts/judge_batch_v1.py prepare "${args[@]}" 2>&1 | tee "$LOG"
    "$PYTHON_BIN" -u scripts/judge_batch_v1.py run "${args[@]}" --allow-network --deployment-tag "$DEPLOYMENT_TAG" --splits train val test --max-new-tasks 48398 --max-http-attempts 96796 2>&1 | tee -a "$LOG"
    "$PYTHON_BIN" -u scripts/judge_batch_v1.py label "${args[@]}" 2>&1 | tee -a "$LOG"
    "$PYTHON_BIN" -u scripts/judge_batch_v1.py validate "${args[@]}" 2>&1 | tee -a "$LOG"
    ;;
  label)
    "$PYTHON_BIN" -u scripts/judge_batch_v1.py label "${args[@]}" 2>&1 | tee "$LOG"
    "$PYTHON_BIN" -u scripts/judge_batch_v1.py validate "${args[@]}" 2>&1 | tee -a "$LOG"
    ;;
  *) echo 'Use MODE=prepare|smoke|full|status|label' >&2;exit 2 ;;
esac
