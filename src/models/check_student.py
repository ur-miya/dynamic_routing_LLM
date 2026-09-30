from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .student_client import StudentClient


def main():
    parser = argparse.ArgumentParser(description="Offline smoke-test for the configured student")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)["student"]
    manifest_path = Path(cfg["local_path"]) / "model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    client = StudentClient(cfg, args.env_file)
    result = client.generate(
        [{"prompt": "Reply with exactly: OK", "history": []}],
        generation_overrides={"do_sample": False, "max_new_tokens": 16},
    )[0]
    if not result.text:
        raise RuntimeError("Student returned an empty response")
    print(
        json.dumps(
            {"model": client.info(), **result.to_dict()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
