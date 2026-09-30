import hashlib, json, sqlite3
from pathlib import Path
from typing import Any, Optional

class SQLiteCache:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")

    @staticmethod
    def key(payload: Any) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        with sqlite3.connect(self.path) as c:
            row = c.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: Any) -> None:
        raw = json.dumps(value, ensure_ascii=False)
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT OR REPLACE INTO cache(key,value) VALUES(?,?)", (key, raw))
