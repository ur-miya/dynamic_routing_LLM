#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
MODE="${MODE:-check}"
case "$MODE" in
 check)
  python3 -m py_compile scripts/rescore_trainval_v3.py scripts/verify_rescore_trainval_v3.py
  python3 -m pytest -q tests/test_rescore_trainval_v3.py tests/test_scoring_v3_core.py
  ;;
 smoke) python3 -u scripts/rescore_trainval_v3.py --smoke "$@" ;;
 full|resume) python3 -u scripts/rescore_trainval_v3.py "$@" ;;
 verify) python3 scripts/verify_rescore_trainval_v3.py "$@" ;;
 *) echo "MODE: check | smoke | full | resume | verify"; exit 2 ;;
esac
