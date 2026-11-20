#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
MODE="${MODE:-check}"
case "$MODE" in
 check)
  python3 -m py_compile scripts/probe_context_tokenizer_v1.py
  python3 -m pytest -q tests/test_context_tokenizer_v1.py
  ;;
 sample) python3 -u scripts/probe_context_tokenizer_v1.py "$@" ;;
 full) python3 -u scripts/probe_context_tokenizer_v1.py --all-val "$@" ;;
 *) echo "MODE: check | sample | full"; exit 2 ;;
esac
