#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:-all}"
PYTHON="${PYTHON:-python3}"

LR_BE_MODEL="artifacts/router/first_pool_v1/run_20261105T192319_497163Z/LR_BE/model.joblib"
LR_BEU_MODEL="artifacts/router/second_pool_v1/run_20261110T194454_001212Z/LR_BEU/model.joblib"

for path in \
    scripts/train_router_v1.py \
    scripts/train_router_pool2_v1.py \
    scripts/experiment_pregen_regression_v1.py \
    configs/router_pool2_v1.yaml \
    "$LR_BE_MODEL" \
    "$LR_BEU_MODEL"
do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path"
        exit 2
    fi
done

run_experiment() {
    "$PYTHON" -u scripts/experiment_pregen_regression_v1.py \
        --config configs/router_pool2_v1.yaml \
        --lr-be-model "$LR_BE_MODEL" \
        --lr-beu-model "$LR_BEU_MODEL" \
        "$@"
}

case "$MODE" in
    check)
        "$PYTHON" -m py_compile \
            scripts/experiment_pregen_regression_v1.py
        "$PYTHON" -m pytest -q \
            tests/test_pregen_regression_v1.py
        ;;
    smoke)
        run_experiment --smoke
        ;;
    full)
        run_experiment
        ;;
    all)
        "$PYTHON" -m py_compile \
            scripts/experiment_pregen_regression_v1.py
        "$PYTHON" -m pytest -q \
            tests/test_pregen_regression_v1.py
        run_experiment --smoke
        run_experiment
        ;;
    *)
        echo "Allowed MODE: check, smoke, full, all"
        exit 2
        ;;
esac