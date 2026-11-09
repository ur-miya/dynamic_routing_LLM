#!/usr/bin/env bash
set -euo pipefail
args=(--input-glob "${INPUT_GLOB:-data/cascade/*_logs.jsonl}" --output "${OUTPUT:-data/cascade/aggregated.parquet}" --batch-size "${BATCH_SIZE:-1024}")
[[ "${OVERWRITE:-0}" == "1" ]] && args+=(--overwrite)
exec python3 -m src.generation.aggregate_cascade "${args[@]}"
