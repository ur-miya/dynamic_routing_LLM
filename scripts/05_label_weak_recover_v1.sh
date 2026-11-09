#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
MODE="${MODE:-smoke}"
case "$MODE" in
  smoke)
    "$PYTHON" scripts/judge_batch_recover_v1.py run --allow-network --max-new-tasks 8 --max-http-attempts 8
    ;;
  full)
    "$PYTHON" scripts/judge_batch_recover_v1.py run --allow-network --max-new-tasks 50000 --max-http-attempts 96612
    "$PYTHON" scripts/judge_batch_recover_v1.py finalize
    ;;
  status)
    "$PYTHON" scripts/judge_batch_recover_v1.py status
    ;;
  *) printf 'MODE must be smoke, full or status\n' >&2; exit 2 ;;
esac
