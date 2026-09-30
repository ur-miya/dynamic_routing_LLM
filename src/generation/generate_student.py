from __future__ import annotations

import argparse
import logging

import yaml

from src.generation.io_utils import ids, messages, read_jsonl, write_parquet
from src.models.student_client import StudentClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--generation", default="configs/generation.yaml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--split", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    with open(args.models, encoding="utf-8") as fh:
        model_config = yaml.safe_load(fh)
    with open(args.generation, encoding="utf-8") as fh:
        generation_config = yaml.safe_load(fh)
    cfg = generation_config["student"]
    client = StudentClient(model_config["student"], args.env_file)
    rows = list(
        read_jsonl(
            generation_config["input_pattern"].format(split=args.split),
            args.limit,
        )
    )
    output = []
    batch_size = int(cfg.get("batch_size", 1))
    overrides = {
        key: cfg[key]
        for key in (
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "do_sample",
            "seed",
        )
        if key in cfg
    }

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        message_batch = [
            messages(
                row,
                generation_config.get("include_history", True),
                generation_config.get("system_prompt"),
            )
            for row in batch
        ]
        try:
            generated = client.generate(
                message_batch,
                max_input_tokens=cfg.get("max_input_tokens"),
                n_samples=int(cfg.get("n_samples", 1)),
                generation_overrides=overrides,
            )
            for row, message_list, result in zip(batch, message_batch, generated):
                logprobs = (
                    client.selected_logprobs(
                        message_list,
                        result.text,
                        int(cfg["max_input_tokens"]) + int(cfg["max_new_tokens"]),
                    )
                    if cfg.get("save_selected_logprobs", False)
                    else []
                )
                output.append(
                    {
                        **ids(row),
                        "SGO": result.text,
                        "SGO_samples": result.samples,
                        "student_output_token_ids": result.output_token_ids,
                        "student_selected_logprobs_sgo": logprobs,
                        "student_latency_ms": result.latency_ms,
                        "student_input_tokens": result.input_tokens,
                        "student_output_tokens": result.output_tokens,
                        "student_tokens_per_second": result.tokens_per_second,
                        "student_batch_size": result.batch_size,
                        "student_finish_reason": result.finish_reason,
                        "student_rendered_prompt": result.rendered_prompt,
                        "student_model_id": client.model_id,
                        "student_source_model": client.config["source_model"],
                        "student_revision": client.config.get("revision"),
                        "student_adapter": client.adapter,
                        "student_status": "ok",
                    }
                )
        except Exception as exc:
            logging.exception("student batch failed")
            for row in batch:
                output.append(
                    {
                        **ids(row),
                        "SGO": "",
                        "SGO_samples": [],
                        "student_output_token_ids": [],
                        "student_selected_logprobs_sgo": [],
                        "student_latency_ms": None,
                        "student_input_tokens": None,
                        "student_output_tokens": None,
                        "student_tokens_per_second": None,
                        "student_batch_size": len(batch),
                        "student_finish_reason": None,
                        "student_rendered_prompt": None,
                        "student_model_id": client.model_id,
                        "student_source_model": client.config["source_model"],
                        "student_revision": client.config.get("revision"),
                        "student_adapter": client.adapter,
                        "student_status": "error",
                        "student_error": str(exc),
                    }
                )
        logging.info("student %s: %d/%d", args.split, len(output), len(rows))

    path = f"{generation_config['work_dir']}/student_{args.split}.parquet"
    write_parquet(output, path)
    print(path)


if __name__ == "__main__":
    main()
