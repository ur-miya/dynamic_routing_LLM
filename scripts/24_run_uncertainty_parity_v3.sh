#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODE="${MODE:-check}"
case "$MODE" in
 check)
  python3 -m py_compile src/features/runtime_uncertainty_v3.py scripts/check_uncertainty_parity_v3.py
  python3 -m pytest -q tests/test_runtime_uncertainty_v3.py
  ;;
 smoke) python3 -u scripts/check_uncertainty_parity_v3.py --smoke "$@" ;;
 full|resume) python3 -u scripts/check_uncertainty_parity_v3.py "$@" ;;
 *) echo "MODE: check | smoke | full | resume"; exit 2 ;;
esac
