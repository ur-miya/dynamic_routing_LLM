from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CascadeLogger:
    def __init__(self, directory: str, filename_pattern: str, fsync: bool = True):
        self.directory = Path(directory)
        self.filename_pattern = filename_pattern
        self.fsync = fsync
        self.directory.mkdir(parents=True, exist_ok=True)

    def current_path(self) -> Path:
        return self.directory / datetime.now(timezone.utc).strftime(self.filename_pattern)

    def write(self, record: Mapping[str, Any]) -> Path:
        payload = dict(record)
        payload.setdefault("logged_at", utc_now())
        encoded = (json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        path = self.current_path()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, encoded)
            if self.fsync:
                os.fsync(fd)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        return path
