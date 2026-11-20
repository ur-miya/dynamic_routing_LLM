#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:-all}"
PYTHON="${PYTHON:-python3}"
CONFIG="configs/router_pool2_v1.yaml"
OUT_ROOT="artifacts/router/second_pool_v1"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

case "$MODE" in
    check|smoke|full|report|all) ;;
    *)
        echo "Unknown MODE=$MODE"
        echo "Allowed: check, smoke, full, report, all"
        exit 2
        ;;
esac

for path in \
    scripts/train_router_v1.py \
    scripts/train_router_pool2_v1.py \
    configs/router_pool2_v1.yaml \
    tests/test_router_pool2_v1.py
do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path"
        exit 2
    fi
done

check_dependencies() {
    "$PYTHON" - <<'PY'
import importlib
import inspect

modules = [
    "numpy",
    "pandas",
    "pyarrow",
    "sklearn",
    "joblib",
    "yaml",
    "threadpoolctl",
    "pytest",
]

for name in modules:
    importlib.import_module(name)

import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

if "class_weight" not in inspect.signature(
    HistGradientBoostingClassifier
).parameters:
    raise RuntimeError(
        "Installed scikit-learn does not support HGB class_weight"
    )

print("DEPENDENCIES: OK")
print("SKLEARN:", sklearn.__version__)
PY
}

verify_pointer() {
    local kind="$1"
    local minimum_mtime="$2"

    "$PYTHON" - "$kind" "$minimum_mtime" "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

kind = sys.argv[1]
minimum_mtime = float(sys.argv[2])
cfg = yaml.safe_load(Path(sys.argv[3]).read_text())

out_root = Path(cfg["output_root"])
pointer = out_root / (
    "latest_smoke.txt" if kind == "smoke" else "latest_run.txt"
)

if not pointer.is_file():
    raise RuntimeError(f"Missing pointer: {pointer}")

if pointer.stat().st_mtime < minimum_mtime:
    raise RuntimeError("Pointer is stale: current run did not publish results")

root = Path(pointer.read_text().strip())
manifest = json.loads((root / "manifest.json").read_text())

expected_smoke = kind == "smoke"
expected_candidates = 6 if expected_smoke else 27
expected_rows = (
    {"train": 120, "val": 120}
    if expected_smoke
    else {"train": 15317, "val": 1847}
)

if manifest["status"] != "complete":
    raise RuntimeError("Run is not complete")

if manifest["smoke"] != expected_smoke:
    raise RuntimeError("Unexpected smoke/full flag")

if manifest["test_read_during_training"] is not False:
    raise RuntimeError("Unexpected test access flag")

if manifest["used_rows"] != expected_rows:
    raise RuntimeError(
        f"Unexpected rows: {manifest['used_rows']}"
    )

if manifest["candidate_count"] != expected_candidates:
    raise RuntimeError(
        f"Unexpected candidate count: {manifest['candidate_count']}"
    )

required_families = {
    "R0_prior",
    "LR_BU",
    "LR_BEU",
    "HGB_B",
    "HGB_BU",
    "MLP_BE",
}

leaderboard = pd.read_csv(root / "leaderboard.csv")

if (
    set(leaderboard["family"]) != required_families
    or len(leaderboard) != len(required_families)
):
    raise RuntimeError("Unexpected selected-model leaderboard")

comparison = pd.read_csv(root / "comparison.csv")

if not expected_smoke:
    expected_comparison = required_families | {"R0", "LR_B", "LR_BE"}

    if (
        set(comparison["family"]) != expected_comparison
        or len(comparison) != len(expected_comparison)
    ):
        raise RuntimeError("First/second pool comparison is incomplete")

for family in required_families:
    for relative in [
        "model.joblib",
        "selection.json",
        "val/metrics.json",
        "val/predictions.parquet",
    ]:
        path = root / family / relative
        if not path.is_file():
            raise RuntimeError(f"Missing artifact: {path}")

numeric = comparison[
    [
        "val_macro_f1",
        "val_balanced_accuracy",
        "val_accuracy",
        "val_log_loss",
        "val_equal_prompt_macro_f1",
    ]
].to_numpy(dtype=float)

import numpy as np

if not np.isfinite(numeric).all():
    raise RuntimeError("Comparison contains non-finite metrics")

if not (root / "reports.zip").is_file():
    raise RuntimeError("Missing reports.zip")

print("VERIFICATION: PASSED")
print("RUN:", root)
print("STATUS:", manifest["status"])
print("SMOKE:", manifest["smoke"])
print("ROWS:", manifest["used_rows"])
print("CANDIDATES:", manifest["candidate_count"])
print("TEST_READ:", manifest["test_read_during_training"])
PY
}

run_tests() {
    check_dependencies
    "$PYTHON" -m py_compile \
        scripts/train_router_v1.py \
        scripts/train_router_pool2_v1.py
    "$PYTHON" -m pytest -q tests/test_router_pool2_v1.py
}

run_training() {
    local kind="$1"
    local started
    started="$("$PYTHON" -c 'import time; print(time.time())')"

    if [[ "$kind" == "smoke" ]]; then
        "$PYTHON" -u scripts/train_router_pool2_v1.py train \
            --config "$CONFIG" \
            --smoke
    else
        "$PYTHON" -u scripts/train_router_pool2_v1.py train \
            --config "$CONFIG"
    fi

    verify_pointer "$kind" "$started"
}

show_report() {
    local run_dir
    if [[ ! -f "$OUT_ROOT/latest_run.txt" ]]; then
        echo "No completed full run found."
        exit 2
    fi

    run_dir="$(cat "$OUT_ROOT/latest_run.txt")"

    "$PYTHON" scripts/train_router_pool2_v1.py report \
        --run-dir "$run_dir"
}

case "$MODE" in
    check)
        run_tests
        ;;
    smoke)
        check_dependencies
        run_training smoke
        ;;
    full)
        check_dependencies
        run_training full
        show_report
        ;;
    report)
        show_report
        ;;
    all)
        run_tests
        run_training smoke
        run_training full
        show_report
        ;;
esac