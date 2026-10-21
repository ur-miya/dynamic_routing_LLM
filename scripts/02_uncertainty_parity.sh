#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd):${PYTHONPATH:-}"
python3 scripts/02_uncertainty_parity.py --split "${SPLIT:-train}" --limit "${LIMIT:-2}" --online-config "${ONLINE_CONFIG:-configs/online_features.yaml}"
