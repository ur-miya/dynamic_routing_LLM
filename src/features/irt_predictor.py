"""Train and apply an out-of-sample prompt-difficulty predictor derived from Rasch IRT."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


class IRTPredictor:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        artifact = joblib.load(self.path)
        if artifact.get("artifact_type") != "irt_embedding_regressor_v1":
            raise ValueError(f"Unsupported IRT artifact: {self.path}")
        self.regressor = artifact["regressor"]
        self.embedding_model = artifact["embedding_model"]
        self.embedding_dimension = int(artifact["embedding_dimension"])
        self.metadata: dict[str, Any] = artifact.get("metadata", {})

    def predict(self, embeddings: np.ndarray, embedding_model: str) -> np.ndarray:
        if embedding_model != self.embedding_model:
            raise ValueError(
                f"IRT artifact expects embedding model {self.embedding_model!r}, got {embedding_model!r}"
            )
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"IRT artifact expects (*, {self.embedding_dimension}), got {embeddings.shape}"
            )
        return np.asarray(self.regressor.predict(embeddings), dtype=np.float32)