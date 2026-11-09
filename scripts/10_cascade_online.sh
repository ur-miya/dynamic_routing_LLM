#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/cascade.yaml}"
PROMPT="${PROMPT:-Explain why HTTPS protects data in transit.}"
HISTORY_JSON="${HISTORY_JSON:-[]}"
FORCE_ROUTE="${FORCE_ROUTE:-auto}"
exec python3 -m src.generation.cascade_inference --config "$CONFIG" --prompt "$PROMPT" --history-json "$HISTORY_JSON" --force-route "$FORCE_ROUTE"
