import argparse
import asyncio
import logging

import pyarrow.parquet as pq
import yaml

from src.models.teacher_client import TeacherClient
from src.generation.io_utils import messages, write_parquet


async def run(args):
    mc = yaml.safe_load(open(args.models))
    gc = yaml.safe_load(open(args.generation))
    cfg = gc["teacher_score"]

    c = TeacherClient(mc["teacher"], args.env_file)
    sem = asyncio.Semaphore(cfg.get("concurrency", 2))

    def row_key(row):
        return (
            str(row["pair_id"]),
            str(row["prompt_id"]),
            str(row["message_tree_id"]),
        )

    teacher = {
        row_key(row): row
        for row in pq.read_table(
            f"{gc['work_dir']}/teacher_{args.split}.parquet"
        ).to_pylist()
    }

    student = {
        row_key(row): row
        for row in pq.read_table(
            f"{gc['work_dir']}/student_{args.split}.parquet"
        ).to_pylist()
    }

    keys = sorted(set(teacher) & set(student))
    if args.limit is not None:
        keys = keys[: args.limit]

    enabled = {str(x).lower() for x in cfg.get("sequences", ["SGO"])}

    async def one(k):
        t, s = teacher[k], student[k]

        # Если промпт нужно брать из student-строки, замените на messages(s)
        prompt_messages = messages(t)

        result = {
            "pair_id": t["pair_id"],
            "prompt_id": t["prompt_id"],
            "message_tree_id": t["message_tree_id"],
        }

        candidates = []

        if "tgo" in enabled:
            candidates.append(("tgo", t.get("TGO", "")))

        if "sgo" in enabled:
            text = str(s.get("SGO") or "").strip()

            if s.get("student_status") != "ok" or not text:
                result["teacher_token_scores_sgo"] = []
                result["teacher_score_sgo_status"] = "skipped"
                result["teacher_score_sgo_error"] = (
                    "Student generation is not successful or SGO is empty"
                )
            else:
                candidates.append(("sgo", text))

        async with sem:
            for name, text in candidates:
                try:
                    x = await c.score_one(
                        prompt_messages,
                        text,
                        cfg.get("top_logprobs", 10),
                    )
                    result[f"teacher_token_scores_{name}"] = x["token_scores"]
                    result[f"teacher_score_{name}_latency_ms"] = x["latency_ms"]
                    result[f"teacher_score_{name}_status"] = "ok"
                except Exception as e:
                    result[f"teacher_token_scores_{name}"] = []
                    result[f"teacher_score_{name}_status"] = "error"
                    result[f"teacher_score_{name}_error"] = str(e)

        return result

    out = []
    shard_size = gc.get("shard_size", 256)

    for start in range(0, len(keys), shard_size):
        out.extend(
            await asyncio.gather(
                *(one(k) for k in keys[start : start + shard_size])
            )
        )
        logging.info("score %s: %d/%d", args.split, len(out), len(keys))

    path = f"{gc['work_dir']}/teacher_scores_{args.split}.parquet"
    write_parquet(out, path)
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--generation", default="configs/generation.yaml")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int)

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()