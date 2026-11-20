#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${MODE:-check}"
case "$MODE" in
 check)
  python3 -m py_compile scripts/experiment_length_ablation_v1.py
  python3 -m pytest -q tests/test_length_ablation_v1.py
  ;;
 smoke) python3 -u scripts/experiment_length_ablation_v1.py --smoke ;;
 full) python3 -u scripts/experiment_length_ablation_v1.py ;;
 *) echo "MODE must be check, smoke or full"; exit 2 ;;
esac
