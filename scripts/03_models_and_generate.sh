#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
MODE="${MODE:-check}"
SPLITS="${SPLITS:-train val test}"
ENV_FILE="${ENV_FILE:-.env}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models.yaml}"
GENERATION_CONFIG="${GENERATION_CONFIG:-configs/generation.yaml}"
LIMIT_ARG=()
[[ -n "${LIMIT:-}" ]] && LIMIT_ARG=(--limit "$LIMIT")
OVERWRITE_ARG=()
[[ -n "${OVERWRITE:-}" ]] && OVERWRITE_ARG=(--overwrite)

case "$MODE" in
  download-student|prepare-student)
    python3 -m src.models.save_model \
      --config "$MODELS_CONFIG" --env-file "$ENV_FILE" "${OVERWRITE_ARG[@]}"
    ;;
  check)
    python3 -m src.models.check_backends \
      --config "$MODELS_CONFIG" --env-file "$ENV_FILE"
    ;;
  check-student)
    python3 -m src.models.check_student \
      --config "$MODELS_CONFIG" --env-file "$ENV_FILE"
    ;;
  teacher)
    for split in $SPLITS; do
      python3 -m src.generation.generate_teacher \
        --models "$MODELS_CONFIG" --generation "$GENERATION_CONFIG" \
        --env-file "$ENV_FILE" --split "$split" "${LIMIT_ARG[@]}"
    done
    ;;
  student)
    for split in $SPLITS; do
      python3 -m src.generation.generate_student \
        --models "$MODELS_CONFIG" --generation "$GENERATION_CONFIG" \
        --env-file "$ENV_FILE" --split "$split" "${LIMIT_ARG[@]}"
    done
    ;;
  score)
    for split in $SPLITS; do
      python3 -m src.generation.score_teacher_sequences \
        --models "$MODELS_CONFIG" --generation "$GENERATION_CONFIG" \
        --env-file "$ENV_FILE" --split "$split" "${LIMIT_ARG[@]}"
    done
    ;;
  merge)
    for split in $SPLITS; do
      python3 -m src.generation.merge_generations \
        --generation "$GENERATION_CONFIG" --split "$split" \
        --include-scores "${OVERWRITE_ARG[@]}"
    done
    ;;
  *)
    echo "MODE: download-student|check|check-student|teacher|student|score|merge" >&2
    exit 2
    ;;
esac
