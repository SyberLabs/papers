"""Parquet-backed cache for normalized ecological tables.

All data adapters (live or sample) write the *same* small set of normalized
tables here. The builder reads exclusively from the cache, which decouples data
acquisition from modeling and makes runs reproducible and offline-capable after
the first fetch.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .logging import get_logger

log = get_logger("cache")


class ParquetCache:
    """A directory of named Parquet tables (plus newick text files)."""

    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str, suffix: str = "parquet") -> Path:
        return self.dir / f"{name}.{suffix}"

    def has(self, name: str, suffix: str = "parquet") -> bool:
        return self.path(name, suffix).exists()

    def write_table(self, name: str, df: pd.DataFrame) -> Path:
        p = self.path(name)
        df.to_parquet(p, index=False)
        log.info("cached table '%s' (%d rows) -> %s", name, len(df), p)
        return p

    def read_table(self, name: str) -> pd.DataFrame:
        p = self.path(name)
        if not p.exists():
            raise FileNotFoundError(f"cache table '{name}' not found at {p}")
        return pd.read_parquet(p)

    def write_text(self, name: str, text: str, suffix: str = "nwk") -> Path:
        p = self.path(name, suffix)
        p.write_text(text, encoding="utf-8")
        log.info("cached text '%s' -> %s", name, p)
        return p

    def read_text(self, name: str, suffix: str = "nwk") -> str:
        p = self.path(name, suffix)
        if not p.exists():
            raise FileNotFoundError(f"cache text '{name}' not found at {p}")
        return p.read_text(encoding="utf-8")
