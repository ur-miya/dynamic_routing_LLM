#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${MODE:-check}"
case "$MODE" in
 check)
  python3 -m py_compile scripts/experiment_hybrid_prompt_sgo_v1.py scripts/verify_hybrid_prompt_sgo_v1.py
  python3 -m pytest -q tests/test_hybrid_prompt_sgo_v1.py tests/test_prompt_sgo_v1.py tests/test_length_ablation_v1.py
  ;;
 smoke) python3 -u scripts/experiment_hybrid_prompt_sgo_v1.py --smoke ;;
 full) python3 -u scripts/experiment_hybrid_prompt_sgo_v1.py ;;
 verify) python3 scripts/verify_hybrid_prompt_sgo_v1.py ;;
 *) echo "MODE: check | smoke | full | verify"; exit 2 ;;
esac
