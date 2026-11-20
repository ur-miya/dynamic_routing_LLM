from __future__ import annotations

import numpy as np


FEATURE_VERSION = "historical-text-rescoring-2"

# Only these columns are passed to the learned gate.
# The length-mismatch diagnostic is deliberately excluded:
# it is not a verified generation-token alignment error.
FEATURES = [
    "sgo_char_count",
    "sgo_word_count",
    "output_token_count",
    "logprob_count",
    "mean_logprob",
    "min_logprob",
    "p10_logprob",
    "std_logprob",
    "mean_probability",
    "geomean_probability",
    "low_probability_fraction",
    "confidence_missing",
    "finish_length",
    "finish_error",
    "empty_answer",
]


def _numeric_logprobs(value):
    if value is None:
        return np.empty(0, dtype=np.float64), False

    try:
        original = np.asarray(value)

        # Reject strings, booleans, objects and non-vector inputs.
        if original.ndim != 1 or original.dtype.kind not in "fiu":
            return np.empty(0, dtype=np.float64), False

        values = original.astype(np.float64, copy=False)

    except (TypeError, ValueError, OverflowError):
        return np.empty(0, dtype=np.float64), False

    valid = bool(
        values.size > 0
        and np.isfinite(values).all()
        and (values <= 1e-6).all()
    )

    if not valid:
        return values, False

    # Treat tiny positive numerical noise as log(1).
    return np.minimum(values, 0.0), True


def _token_count(value):
    if value is None:
        return 0, False

    try:
        ids = np.asarray(value)
        valid = bool(
            ids.ndim == 1
            and ids.size > 0
            and ids.dtype.kind in "iu"
            and (ids >= 0).all()
        )
        return int(ids.size) if ids.ndim == 1 else 0, valid

    except (TypeError, ValueError, OverflowError):
        return 0, False


def extract_student_features(
    text,
    logprobs,
    token_ids,
    finish_reason,
    status="ok",
):
    text_valid = isinstance(text, str)
    text = text if text_valid else ""

    values, confidence_valid = _numeric_logprobs(logprobs)
    token_count, token_ids_valid = _token_count(token_ids)

    finish = str(finish_reason or "").lower()
    status_value = str(status or "").lower()

    empty = not bool(text.strip())
    error = (
        not text_valid
        or status_value != "ok"
        or finish not in {"stop", "eos", "length"}
    )

    # This is only a length diagnostic.
    # Neither equality nor inequality proves token alignment.
    length_mismatch = bool(
        token_ids_valid and token_count != values.size
    )

    result = {
        "sgo_char_count": float(len(text)),
        "sgo_word_count": float(len(text.split())),
        "output_token_count": (
            float(token_count) if token_ids_valid else np.nan
        ),
        "logprob_count": float(values.size),
        "mean_logprob": np.nan,
        "min_logprob": np.nan,
        "p10_logprob": np.nan,
        "std_logprob": np.nan,
        "mean_probability": np.nan,
        "geomean_probability": np.nan,
        "low_probability_fraction": np.nan,
        "confidence_missing": float(not confidence_valid),
        "finish_length": float(finish == "length"),
        "finish_error": float(error),
        "empty_answer": float(empty),

        # Diagnostics: not model inputs.
        "token_logprob_length_mismatch": float(length_mismatch),
        "token_ids_invalid": float(not token_ids_valid),
    }

    if confidence_valid:
        probabilities = np.exp(values)

        result.update({
            "mean_logprob": float(values.mean()),
            "min_logprob": float(values.min()),
            "p10_logprob": float(np.quantile(values, 0.10)),
            "std_logprob": float(values.std()),
            "mean_probability": float(probabilities.mean()),
            "geomean_probability": float(np.exp(values.mean())),
            "low_probability_fraction": float(
                (values < np.log(0.1)).mean()
            ),
        })

    reasons = []

    if error:
        reasons.append("student_error")
    if empty:
        reasons.append("empty_answer")
    if finish == "length":
        reasons.append("length_limit")
    if not confidence_valid:
        reasons.append("confidence_unavailable")

    return result, bool(reasons), "|".join(reasons)