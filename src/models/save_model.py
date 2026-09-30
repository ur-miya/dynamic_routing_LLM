from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["student"]


def main():
    parser = argparse.ArgumentParser(description="Pin and download the base student model once")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=False)
    cfg = load_config(args.config)
    target = Path(cfg["local_path"])
    manifest_path = target / "model_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = (cfg["source_model"], cfg["revision"])
        actual = (manifest.get("source_model"), manifest.get("revision"))
        if actual != expected:
            raise RuntimeError(f"Existing student differs: {actual} != {expected}")
        print(f"Student already prepared: {target}")
        return
    if target.exists() and any(target.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Non-empty target without valid manifest: {target}")
    target.mkdir(parents=True, exist_ok=True)
    token = os.getenv(cfg.get("token_env", "HF_TOKEN")) or None
    info = HfApi(token=token).model_info(cfg["source_model"], revision=cfg["revision"])
    resolved_sha = info.sha
    if resolved_sha != cfg["revision"]:
        raise RuntimeError(f"Revision resolved to unexpected SHA: {resolved_sha}")
    snapshot_download(
        repo_id=cfg["source_model"],
        revision=cfg["revision"],
        local_dir=target,
        token=token,
    )
    manifest = {
        "source_model": cfg["source_model"],
        "revision": cfg["revision"],
        "resolved_sha": resolved_sha,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared student at {target}")


if __name__ == "__main__":
    main()
