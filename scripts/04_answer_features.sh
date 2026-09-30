#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
MODE="${MODE:-basic}"; SPLITS="${SPLITS:-train val test}"; LIMIT_ARG=(); [[ -n "${LIMIT:-}" ]] && LIMIT_ARG=(--limit "$LIMIT")
if [[ "$MODE" == assemble ]]; then
  for s in $SPLITS; do python3 -m src.features.assemble_answer_features --split "$s" --include ${INCLUDE:-reference bertscore judge} ${OVERWRITE:+--overwrite}; done
else
  for s in $SPLITS; do python3 -m src.features.answer_features --mode "$MODE" --split "$s" "${LIMIT_ARG[@]}"; done
fi
