"""Fyers market-data transformations shared by live trading and replay."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from Dependencies.market_data_health import complete_minute_bucket_mask

IST = ZoneInfo("Asia/Kolkata")


def completed_five_minute_bars(minute_bars: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Aggregate only complete five-minute buckets containing all five minutes."""
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(minute_bars.columns):
        raise ValueError("Fyers minute data is missing required OHLCV columns.")
    frame = minute_bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].dt.tz is None:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(IST)
    else:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert(IST)
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    completed_before = pd.Timestamp(now.astimezone(IST)).floor("5min")
    frame = frame[frame.index < completed_before]
    complete = complete_minute_bucket_mask(pd.DatetimeIndex(frame.index), 5)
    frame = frame.loc[complete.to_numpy()]
    bars = frame.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    counts = frame["close"].resample("5min", label="left", closed="left").count()
    bars = bars[(counts == 5) & bars["close"].notna()]
    return bars.reset_index(names="timestamp")
