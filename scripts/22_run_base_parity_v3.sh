#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODE="${MODE:-check}"
case "$MODE" in
 check)
  python3 -m py_compile src/features/runtime_base_v3.py scripts/check_base_parity_v3.py
  python3 -m pytest -q tests/test_runtime_base_v3.py
  ;;
 smoke) python3 -u scripts/check_base_parity_v3.py --smoke "$@" ;;
 full) python3 -u scripts/check_base_parity_v3.py "$@" ;;
 *) echo "MODE: check | smoke | full"; exit 2 ;;
esac
