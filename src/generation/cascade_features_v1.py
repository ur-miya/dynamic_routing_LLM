from __future__ import annotations

import numpy as np


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
    "token_alignment_error",
]


def extract_student_features(
    text,
    logprobs,
    token_ids,
    finish_reason,
    status="ok",
):
    if not isinstance(text, str):
        text = ""

    def sequence(value):
        if value is None:
            return []
        return list(value)

    try:
        raw = sequence(logprobs)
        values = np.asarray(raw, dtype=np.float64)
        valid = (
            values.ndim == 1
            and values.size > 0
            and np.isfinite(values).all()
            and (values <= 1e-6).all()
        )
    except (TypeError, ValueError):
        values = np.array([], dtype=np.float64)
        valid = False

    try:
        ids = sequence(token_ids)
        token_count = len(ids)
    except (TypeError, ValueError):
        token_count = 0

    alignment_error = (
        token_count == 0
        or token_count != len(values)
    )
    valid = bool(valid and not alignment_error)

    finish = str(finish_reason or "").lower()
    status_value = str(status or "").lower()

    empty = not bool(text.strip())
    error = status_value != "ok" or finish not in {
        "stop", "eos", "length"
    }

    result = {
        "sgo_char_count": float(len(text)),
        "sgo_word_count": float(len(text.split())),
        "output_token_count": float(token_count),
        "logprob_count": float(len(values)),
        "mean_logprob": np.nan,
        "min_logprob": np.nan,
        "p10_logprob": np.nan,
        "std_logprob": np.nan,
        "mean_probability": np.nan,
        "geomean_probability": np.nan,
        "low_probability_fraction": np.nan,
        "confidence_missing": float(not valid),
        "finish_length": float(finish == "length"),
        "finish_error": float(error),
        "empty_answer": float(empty),
        "token_alignment_error": float(alignment_error),
    }

    if valid:
        result.update({
            "mean_logprob": float(values.mean()),
            "min_logprob": float(values.min()),
            "p10_logprob": float(np.quantile(values, 0.10)),
            "std_logprob": float(values.std()),
            "mean_probability": float(np.exp(values).mean()),
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
    if not valid:
        reasons.append("confidence_unavailable")

    return result, bool(reasons), "|".join(reasons)