"""In-memory feature assembly and single-request scoring. No dataset I/O."""
from __future__ import annotations

import numpy as np
import pyarrow as pa


DIRECT_METHODS = (
    "direct_LR_BE",
    "direct_LR_BEU",
    "direct_Ridge_BE",
    "direct_Ridge_BEU",
)
FLAG_FIELDS = (
    "finish_error",
    "empty_answer",
    "finish_length",
    "confidence_missing",
)


class RuntimeMatrixV3:
    def __init__(self, bundles, *, base_schema, embedding_schema, uq_schema):
        self.bundles = bundles
        self.base = list(bundles["direct_LR_BE"]["base_columns"])
        self.uq = list(bundles["direct_LR_BEU"]["uncertainty_columns"])
        self.post = list(bundles["cascade_A"]["sgo_features"])
        self.dim = int(bundles["direct_LR_BE"]["embedding_dim"])

        if len(self.base) != 24 or len(self.uq) != 11:
            raise ValueError("Unexpected B/U dimensions")
        if self.dim != 384 or len(self.post) != 15:
            raise ValueError("Unexpected E/SGO dimensions")

        self.base_types = [base_schema.field(name).type for name in self.base]
        self.uq_types = [uq_schema.field(name).type for name in self.uq]
        self.embedding_type = embedding_schema.field("prompt_embedding").type

        if any(dtype != pa.float32() for dtype in self.uq_types):
            raise ValueError("Expected original float32 uncertainty schema")
        if (
            not pa.types.is_fixed_size_list(self.embedding_type)
            or self.embedding_type.list_size != self.dim
            or self.embedding_type.value_type != pa.float32()
        ):
            raise ValueError("Unexpected original embedding schema")

        be_names = self.base + [
            f"prompt_embedding_{i}" for i in range(self.dim)
        ]
        beu_names = be_names + [
            "log1p_student_prompt_perplexity"
            if name == "student_prompt_perplexity" else name
            for name in self.uq
        ]

        for method in DIRECT_METHODS:
            bundle = bundles[method]
            expected = beu_names if method.endswith("_BEU") else be_names
            if "feature_names" in bundle:
                if list(bundle["feature_names"]) != expected:
                    raise ValueError(f"Saved feature order differs: {method}")
            if list(bundle.get("classes", ["easy", "medium", "hard"])) != [
                "easy", "medium", "hard"
            ]:
                raise ValueError(f"Saved classes differ: {method}")

        gate = bundles["cascade_A"]
        if gate["feature_group"] != "PROMPT_SGO":
            raise ValueError("Expected PROMPT_SGO gate")
        if gate["scoring_contract"] != "generation-token-raw-fp32-eager-v3":
            raise ValueError("Unexpected SGO scoring contract")
        if gate["feature_order"] != (
            "SGO FEATURES, then original LR_BE loader X for PROMPT_SGO"
        ):
            raise ValueError("Unexpected gate feature order")

        pipelines = [
            (bundles["direct_LR_BE"]["estimator"], 408),
            (bundles["direct_Ridge_BE"]["model"], 408),
            (bundles["direct_Ridge_BEU"]["model"], 419),
            (gate["model"], 423),
        ]
        members = bundles["direct_LR_BEU"]["models"]
        if not members:
            raise ValueError("Empty LR_BEU model ensemble")
        pipelines.extend((model, 419) for model in members)

        for model, width in pipelines:
            if not hasattr(model, "steps"):
                raise ValueError("Expected saved preprocessing pipeline")
            if model.n_features_in_ != width:
                raise ValueError("Saved pipeline input dimension differs")

    @staticmethod
    def _cast_fields(record, names, types):
        missing = [name for name in names if name not in record]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        values = []
        for name, dtype in zip(names, types):
            value = pa.array([record[name]], type=dtype)[0].as_py()
            values.append(np.nan if value is None else value)
        result = np.asarray(values, dtype=np.float64)
        if np.isinf(result).any():
            raise ValueError("Feature block contains infinity")
        return result

    def assemble_be(self, base, embedding):
        b = self._cast_fields(base, self.base, self.base_types)
        raw = np.asarray(embedding)
        if raw.shape != (self.dim,):
            raise ValueError("Embedding must be a single 384-vector")

        e = np.asarray(
            pa.array([raw.tolist()], type=self.embedding_type)[0].as_py(),
            dtype=np.float64,
        )
        if not np.isfinite(e).all() or not np.any(e):
            raise ValueError("Invalid embedding values")
        return np.concatenate([b, e]).reshape(1, -1)

    def assemble_beu(self, be, uncertainty):
        self._check_row(be, 408)
        u = self._cast_fields(uncertainty, self.uq, self.uq_types)
        if (u[np.isfinite(u)] < 0).any():
            raise ValueError("Negative uncertainty feature")

        for name in ("student_input_truncated", "student_prompt_too_short"):
            value = u[self.uq.index(name)]
            if np.isfinite(value) and value not in (0.0, 1.0):
                raise ValueError(f"Invalid uncertainty flag: {name}")

        i = self.uq.index("student_prompt_perplexity")
        u[i] = np.log1p(u[i])
        return np.concatenate([be[0], u]).reshape(1, -1)

    def assemble_post(self, be, post):
        self._check_row(be, 408)
        missing = [name for name in self.post if name not in post]
        if missing:
            raise ValueError(f"Missing SGO features: {missing}")
        values = np.asarray([post[name] for name in self.post], dtype=np.float64)
        if np.isinf(values).any():
            raise ValueError("SGO features contain infinity")
        return np.concatenate([values, be[0]]).reshape(1, -1)

    @staticmethod
    def _check_row(row, width):
        if row.shape != (1, width) or row.dtype != np.dtype("float64"):
            raise ValueError("Expected single float64 feature row")

    @staticmethod
    def _probabilities(model, row):
        raw = model.predict_proba(row)
        result = np.zeros((1, 3), dtype=np.float64)
        for column, class_id in enumerate(model.classes_):
            index = int(class_id)
            if index not in (0, 1, 2):
                raise ValueError("Unexpected class ID")
            result[:, index] = raw[:, column]
        if (
            not np.isfinite(result).all()
            or (result < 0).any()
            or not np.allclose(result.sum(axis=1), 1.0)
        ):
            raise ValueError("Invalid predicted probabilities")
        return result

    def score_direct(self, method, row):
        width = 419 if method.endswith("_BEU") else 408
        self._check_row(row, width)
        bundle = self.bundles[method]

        if method == "direct_LR_BE":
            value = self._probabilities(bundle["estimator"], row)[0, 2]
        elif method == "direct_LR_BEU":
            probabilities = np.mean([
                self._probabilities(model, row)
                for model in bundle["models"]
            ], axis=0)
            if not np.allclose(probabilities.sum(axis=1), 1.0):
                raise ValueError("Invalid ensemble probabilities")
            value = probabilities[0, 2]
        elif method in ("direct_Ridge_BE", "direct_Ridge_BEU"):
            value = bundle["model"].predict(row)[0]
        else:
            raise ValueError(f"Unsupported direct method: {method}")

        if not np.isfinite(value):
            raise ValueError("Nonfinite direct score")
        return float(value)

    def score_gate(self, row):
        self._check_row(row, 423)
        value = self.bundles["cascade_A"]["model"].predict(row)[0]
        if not np.isfinite(value):
            raise ValueError("Nonfinite gate score")
        return float(value)
