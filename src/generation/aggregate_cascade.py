from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def flatten(record: dict) -> dict:
    usage = record.get("teacher_usage") or {}
    return {
        "schema_version": record.get("schema_version"),
        "request_id": record.get("request_id"),
        "started_at": record.get("started_at"),
        "logged_at": record.get("logged_at"),
        "prompt_sha256": record.get("prompt_sha256"),
        "prompt": record.get("prompt"),
        "history_json": json.dumps(record.get("history") or [], ensure_ascii=False),
        "SGO": record.get("SGO"),
        "confidence": record.get("confidence"),
        "confidence_metric": record.get("confidence_metric"),
        "student_mean_logprob": record.get("student_mean_logprob"),
        "student_mean_entropy": record.get("student_mean_entropy"),
        "student_finish_reason": record.get("student_finish_reason"),
        "student_latency_ms": record.get("student_latency_ms"),
        "student_input_tokens": record.get("student_input_tokens"),
        "student_output_tokens": record.get("student_output_tokens"),
        "threshold": record.get("threshold"),
        "route": record.get("route"),
        "route_reason": record.get("route_reason"),
        "escalated": record.get("escalated"),
        "TGO": record.get("TGO"),
        "teacher_finish_reason": record.get("teacher_finish_reason"),
        "teacher_latency_ms": record.get("teacher_latency_ms"),
        "teacher_prompt_tokens": usage.get("prompt_tokens"),
        "teacher_completion_tokens": usage.get("completion_tokens"),
        "teacher_total_tokens": usage.get("total_tokens"),
        "total_latency_ms": record.get("total_latency_ms"),
        "status": record.get("status"),
        "error": record.get("error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="data/cascade/*_logs.jsonl")
    parser.add_argument("--output", default="data/cascade/aggregated.parquet")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    writer = None
    batch: list[dict] = []
    seen: set[str] = set()
    rows = 0
    try:
        for path in sorted(Path().glob(args.input_glob)):
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                    request_id = record.get("request_id")
                    if not request_id or request_id in seen:
                        continue
                    seen.add(request_id)
                    batch.append(flatten(record))
                    if len(batch) >= args.batch_size:
                        table = pa.Table.from_pylist(batch)
                        writer = writer or pq.ParquetWriter(tmp, table.schema, compression="zstd")
                        writer.write_table(table)
                        rows += len(batch)
                        batch.clear()
        if batch:
            table = pa.Table.from_pylist(batch)
            writer = writer or pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(batch)
        if writer is None:
            raise RuntimeError("No cascade records found")
        writer.close()
        writer = None
        tmp.replace(output)
    finally:
        if writer is not None:
            writer.close()
        if tmp.exists():
            tmp.unlink()
    print(json.dumps({"output": str(output), "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
