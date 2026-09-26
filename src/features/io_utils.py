"""Streaming JSONL and Arrow/Parquet helpers."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

KEY_FIELDS = ("pair_id", "prompt_id", "message_tree_id")


def iter_jsonl(path: Path, chunk_size: int, limit: int | None = None) -> Iterator[list[dict[str, Any]]]:
    chunk: list[dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            missing = [field for field in (*KEY_FIELDS, "prompt", "history") if field not in row]
            if missing:
                raise ValueError(f"Missing fields {missing} at {path}:{line_number}")
            chunk.append(row)
            seen += 1
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
            if limit is not None and seen >= limit:
                break
    if chunk:
        yield chunk


def key_columns(rows: list[dict[str, Any]]) -> dict[str, pa.Array]:
    return {field: pa.array([str(row[field]) for row in rows], type=pa.string()) for field in KEY_FIELDS}


def fixed_vector_array(values: np.ndarray) -> pa.FixedSizeListArray:
    if values.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {values.shape}")
    values = np.ascontiguousarray(values, dtype=np.float32)
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


class AtomicParquetWriter:
    def __init__(self, output: Path, compression: str = "zstd", overwrite: bool = False):
        self.output = output
        if output.exists() and not overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
        os.close(fd)
        self.temp = Path(name)
        self.compression = compression
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0

    def write(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.temp, table.schema, compression=self.compression)
        elif table.schema != self.writer.schema:
            raise ValueError("Parquet schema changed between chunks")
        self.writer.write_table(table, row_group_size=table.num_rows)
        self.rows += table.num_rows

    def close(self) -> None:
        if self.writer is None:
            raise ValueError("No rows were produced")
        self.writer.close()
        os.replace(self.temp, self.output)

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self.temp.unlink(missing_ok=True)
