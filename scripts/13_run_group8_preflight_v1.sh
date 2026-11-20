#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${MODE:-check}"
case "$MODE" in
 check)
  python3 -m py_compile scripts/preflight_group8_v1.py
  python3 -m pytest -q tests/test_preflight_group8_v1.py
  ;;
 audit) python3 -u scripts/preflight_group8_v1.py "$@" ;;
 *) echo "MODE: check | audit"; exit 2 ;;
esac
