"""Persistent cache for immutable Fyers historical candle requests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from Dependencies.fyers_api import FyersAPI

IST = ZoneInfo("Asia/Kolkata")
HISTORY_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
CACHE_VERSION = 1


class FyersHistoryCache:
    """Cache historical candle requests on disk and report hit/miss counts."""

    def __init__(self, cache_dir: Path, *, enabled: bool = True, refresh: bool = False) -> None:
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.refresh = refresh
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(
        symbol: str,
        resolution: str,
        start: date,
        end: date,
        expired: bool,
    ) -> str:
        identity = json.dumps(
            {
                "version": CACHE_VERSION,
                "symbol": symbol,
                "resolution": resolution,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "expired": expired,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def history(
        self,
        api: FyersAPI,
        symbol: str,
        *,
        resolution: str,
        start: date,
        end: date,
        expired: bool = False,
    ) -> pd.DataFrame:
        if not self.enabled:
            self.misses += 1
            return api.history(
                symbol,
                resolution=resolution,
                start=start,
                end=end,
                expired=expired,
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{self._key(symbol, resolution, start, end, expired)}.csv"
        if path.is_file() and not self.refresh:
            try:
                cached = pd.read_csv(path)
                if list(cached.columns) != HISTORY_COLUMNS:
                    raise ValueError("unexpected cached candle columns")
                if not cached.empty:
                    cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True).dt.tz_convert(IST)
                    for column in HISTORY_COLUMNS[1:]:
                        cached[column] = pd.to_numeric(cached[column], errors="raise")
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                raise RuntimeError(
                    f"Unable to read Fyers history cache {path}. Clear that cache file."
                ) from exc
            self.hits += 1
            return cached[HISTORY_COLUMNS]

        self.misses += 1
        history = api.history(
            symbol,
            resolution=resolution,
            start=start,
            end=end,
            expired=expired,
        )
        if list(history.columns) != HISTORY_COLUMNS:
            raise ValueError("Fyers history returned an unexpected candle schema; cache was not written.")
        normalized = history[HISTORY_COLUMNS].copy()
        if not normalized.empty:
            timestamps = pd.to_datetime(normalized["timestamp"], errors="raise")
            normalized["timestamp"] = (
                timestamps.dt.tz_localize(IST) if timestamps.dt.tz is None else timestamps.dt.tz_convert(IST)
            )
            for column in HISTORY_COLUMNS[1:]:
                normalized[column] = pd.to_numeric(normalized[column], errors="raise")

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=self.cache_dir,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            normalized.to_csv(temporary_path, index=False)
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return normalized
