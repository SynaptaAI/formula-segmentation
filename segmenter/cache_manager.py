import os
import sqlite3
import hashlib
import json
from typing import Optional, Any, Dict

class DiskCache:
    """
    Simple SQLite-based disk cache for LLM responses.
    """
    def __init__(self, cache_dir: str = ".cache", db_name: str = "llm_cache.db"):
        self.cache_dir = cache_dir
        self.db_path = os.path.join(cache_dir, db_name)
        self._init_db()

    def _init_db(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                timestamp REAL
            )
        ''')
        conn.commit()
        conn.close()

    def get(self, key: str) -> Optional[Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT value FROM cache WHERE key = ?", (key,))
            row = c.fetchone()
            conn.close()
            
            if row:
                return json.loads(row[0])
            return None
        except Exception as e:
            print(f"Cache read error: {e}")
            return None

    def set(self, key: str, value: Any):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            json_val = json.dumps(value)
            c.execute(
                "INSERT OR REPLACE INTO cache (key, value, timestamp) VALUES (?, ?, strftime('%s', 'now'))",
                (key, json_val)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Cache write error: {e}")

    def generate_key(self, prompt: str, model: str, params: Dict[str, Any] = None) -> str:
        """Generate a deterministic hash key."""
        params_str = json.dumps(params or {}, sort_keys=True)
        raw = f"{model}::{params_str}::{prompt}"
        return hashlib.md5(raw.encode()).hexdigest()
