#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
MODE="${MODE:-check}"
case "$MODE" in
 check)
  python3 -m py_compile scripts/robustness_corrected_v3.py scripts/verify_robustness_corrected_v3.py
  python3 -m pytest -q tests/test_robustness_corrected_v3.py
  ;;
 smoke) python3 -u scripts/robustness_corrected_v3.py --smoke "$@" ;;
 full) python3 -u scripts/robustness_corrected_v3.py "$@" ;;
 verify) python3 scripts/verify_robustness_corrected_v3.py ;;
 *) echo "MODE: check | smoke | full | verify"; exit 2 ;;
esac
