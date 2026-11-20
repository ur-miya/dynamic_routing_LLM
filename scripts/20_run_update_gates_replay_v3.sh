#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${MODE:-check}"
case "$MODE" in
 check)
  python3 -m py_compile scripts/update_gates_replay_v3.py scripts/verify_gates_replay_v3.py
  python3 -m pytest -q tests/test_gates_replay_v3.py
  ;;
 smoke) python3 -u scripts/update_gates_replay_v3.py --smoke "$@" ;;
 full) python3 -u scripts/update_gates_replay_v3.py "$@" ;;
 verify) python3 scripts/verify_gates_replay_v3.py ;;
 *) echo "MODE: check | smoke | full | verify"; exit 2 ;;
esac
