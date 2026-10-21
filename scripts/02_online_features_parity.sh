#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-configs/online_features.yaml}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-2}"
"$PYTHON_BIN" -m src.features.validate_online_parity --config "$CONFIG" --split "$SPLIT" --limit "$LIMIT"