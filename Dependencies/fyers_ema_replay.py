"""Replay the live Fyers EMA process against historical index and option candles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Dependencies.fyers_api import FyersAPI, FyersAPIError  # noqa: E402
from Dependencies.fyers_market_data import completed_five_minute_bars  # noqa: E402
from Dependencies.fyers_options import (  # noqa: E402
    FyersOptionContract,
    _parse_expiry,
    select_atm_option,
    select_next_week_expiry,
)
from Dependencies.fyers_telegram_auth import load_persisted_access_token  # noqa: E402

ENV_FILE = ROOT_DIR / "Dependencies" / ".env"
OUTPUT_DIR = ROOT_DIR / "Backtest Outputs"
IST = ZoneInfo("Asia/Kolkata")
UNDERLYING_SYMBOL = "NSE:NIFTY50-INDEX"
INDEX_SYMBOLS = {
    "nifty": ("NIFTY", "NSE:NIFTY50-INDEX"),
    "sensex": ("SENSEX", "BSE:SENSEX-INDEX"),
}
MARKET_OPEN = time(9, 15)
ENTRY_START = time(9, 25)
SQUARE_OFF = time(15, 15)
MAX_ENTRY_BAR_AGE = timedelta(minutes=2)
HISTORY_CHUNK_DAYS = 90
OPTION_RESOLUTION_SECONDS = {"5S": 5, "1": 60}
INDEX_TRAILING_STOPS = (
    "none",
    "swing_2bar",
    "atr_2x",
    "breakeven_1r_ema9",
    "breakeven_1r_ema11",
    "combined",
)
HISTORY_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
CACHE_VERSION = 1


@dataclass(slots=True)
class _OptionQuotes:
    completed_at_ns: np.ndarray
    close: np.ndarray

    def at(self, timestamp: pd.Timestamp) -> tuple[float, pd.Timestamp] | None:
        index = int(np.searchsorted(self.completed_at_ns, timestamp.value, side="right")) - 1
        if index < 0:
            return None
        price = float(self.close[index])
        if not math.isfinite(price) or price <= 0:
            return None
        return price, pd.Timestamp(self.completed_at_ns[index], tz=IST)


@dataclass(slots=True)
class _Position:
    contract: FyersOptionContract
    action: str
    direction: str
    quantity: int
    signal_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp
    entry_quote_timestamp: pd.Timestamp
    entry_spot: float
    entry_option: float
    option_quotes: _OptionQuotes


def _index_history_warmup_start(start: date, today: date) -> date:
    warmup_start = start - timedelta(days=7)
    intraday_history_start = today - timedelta(days=365)
    if start >= intraday_history_start:
        return max(warmup_start, intraday_history_start)
    return warmup_start


def _load_signal_logic() -> Any:
    path = ROOT_DIR / "Signal Generators" / "ema_trend_strategy_logic.py"
    spec = importlib.util.spec_from_file_location("fyers_replay_ema_logic", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load EMA strategy logic from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _history_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=HISTORY_CHUNK_DAYS - 1), end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks


def _expiry_lookup_end(end: date, today: date) -> date:
    """Include next-week expiries needed by signals near the end of the replay."""
    return min(end + timedelta(days=7), today - timedelta(days=1))


def _parse_cache_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("useCache must be true or false.")


class FyersHistoryCache:
    """Persistent cache of immutable historical candle requests."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        enabled: bool = True,
        refresh: bool = False,
    ) -> None:
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
                raise RuntimeError(f"Unable to read Fyers history cache {path}. Clear that cache file.") from exc
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
        history = self._normalize(history)
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
            history.to_csv(temporary_path, index=False)
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return history.copy()

    @staticmethod
    def _normalize(history: pd.DataFrame) -> pd.DataFrame:
        normalized = history[HISTORY_COLUMNS].copy()
        if not normalized.empty:
            timestamps = pd.to_datetime(normalized["timestamp"], errors="raise")
            timestamps = timestamps.dt.tz_localize(IST) if timestamps.dt.tz is None else timestamps.dt.tz_convert(IST)
            normalized["timestamp"] = timestamps
            for column in HISTORY_COLUMNS[1:]:
                normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        return normalized


def _fetch_index_history(
    api: FyersAPI,
    start: date,
    end: date,
    history_cache: FyersHistoryCache | None = None,
    *,
    symbol: str = UNDERLYING_SYMBOL,
) -> pd.DataFrame:
    frames = []
    day = start
    while day <= end:
        if history_cache is None:
            frame = api.history(symbol, resolution="1", start=day, end=day)
        else:
            frame = history_cache.history(
                api,
                symbol,
                resolution="1",
                start=day,
                end=day,
            )
        frames.append(frame)
        day += timedelta(days=1)
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def _active_expiry(row: dict[str, Any]) -> date | None:
    try:
        return _parse_expiry(row.get("expiry"))
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _archived_option_rows(symbols: list[str], expiry: date, lot_size: int) -> list[dict[str, Any]]:
    month = expiry.strftime("%b").upper()
    prefixes = (
        f"NIFTY{expiry:%y}{expiry.month}{expiry:%d}",
        f"NIFTY{expiry:%y}{month[0]}{expiry:%d}",
        f"NIFTY{expiry:%y}{month}",
    )
    rows = []
    for symbol in symbols:
        name = str(symbol).strip().upper().rsplit(":", 1)[-1]
        option_type = name[-2:] if name.endswith(("CE", "PE")) else ""
        prefix = next((candidate for candidate in prefixes if name.startswith(candidate)), None)
        if prefix is None or option_type not in {"CE", "PE"}:
            continue
        strike_text = name[len(prefix) : -2]
        if not strike_text.isdigit():
            continue
        rows.append(
            {
                "symbol": symbol,
                "option_type": option_type,
                "strike_price": float(strike_text),
                "lot_size": lot_size,
                "expiry": expiry,
            }
        )
    return rows


def _resolve_contract(
    api: FyersAPI,
    *,
    action: str,
    spot: float,
    trade_date: date,
    active_rows: list[dict[str, Any]],
    expiries: list[date],
    archived_symbols: dict[date, list[str]],
    fallback_lot_size: int,
) -> FyersOptionContract:
    expiry = select_next_week_expiry(expiries, today=trade_date)
    if expiry < date.today():
        if expiry not in archived_symbols:
            archived_symbols[expiry] = api.expired_option_symbols(UNDERLYING_SYMBOL, expiry)
        rows = _archived_option_rows(archived_symbols[expiry], expiry, fallback_lot_size)
    else:
        rows = [
            {**row, "expiry": expiry}
            for row in active_rows
            if row.get("underlying") == "NIFTY"
            and row.get("option_type") in {"CE", "PE"}
            and _active_expiry(row) == expiry
        ]
    return select_atm_option(rows, action=action, spot=spot, expiry=expiry)


def _quote_at(
    option_bars: pd.DataFrame | _OptionQuotes,
    at: pd.Timestamp,
    resolution: str,
) -> tuple[float, pd.Timestamp] | None:
    if isinstance(option_bars, _OptionQuotes):
        return option_bars.at(at)
    seconds = OPTION_RESOLUTION_SECONDS[resolution]
    bars = option_bars
    if bars.empty:
        return None
    timestamps = pd.to_datetime(bars["timestamp"], errors="coerce")
    timestamps = timestamps.dt.tz_localize(IST) if timestamps.dt.tz is None else timestamps.dt.tz_convert(IST)
    completed = bars.loc[timestamps + pd.to_timedelta(seconds, unit="s") <= at].copy()
    if completed.empty:
        return None
    completed["timestamp"] = timestamps.loc[completed.index]
    completed = completed.sort_values("timestamp")
    last = completed.iloc[-1]
    try:
        price = float(last["close"])
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(price) or price <= 0:
        return None
    return price, pd.Timestamp(last["timestamp"]) + pd.Timedelta(seconds=seconds)


def _prepare_option_quotes(option_bars: pd.DataFrame, resolution: str) -> _OptionQuotes:
    seconds = OPTION_RESOLUTION_SECONDS[resolution]
    if option_bars.empty:
        return _OptionQuotes(np.array([], dtype=np.int64), np.array([], dtype=np.float64))
    timestamps = pd.to_datetime(option_bars["timestamp"], errors="coerce")
    timestamps = timestamps.dt.tz_localize(IST) if timestamps.dt.tz is None else timestamps.dt.tz_convert(IST)
    prepared = pd.DataFrame(
        {
            "completed_at": timestamps + pd.Timedelta(seconds=seconds),
            "close": pd.to_numeric(option_bars["close"], errors="coerce"),
        }
    ).dropna(subset=["completed_at"])
    prepared = prepared.sort_values("completed_at").drop_duplicates("completed_at", keep="last")
    completed_at = prepared["completed_at"].dt.as_unit("ns")
    return _OptionQuotes(
        completed_at.array.asi8.copy(),
        prepared["close"].to_numpy(dtype=np.float64, copy=True),
    )


def _action_polls(day: date, poll_seconds: int) -> list[pd.Timestamp]:
    start = pd.Timestamp(datetime.combine(day, MARKET_OPEN), tz=IST)
    end = pd.Timestamp(datetime.combine(day, SQUARE_OFF), tz=IST) + pd.Timedelta(seconds=poll_seconds)
    return list(pd.date_range(start, end, freq=pd.Timedelta(seconds=poll_seconds)))


def replay_index_only(
    *,
    signal_logic: Any,
    strategy_config: Any,
    index_bars: pd.DataFrame,
    index_name: str = "NIFTY",
    start: date,
    end: date,
    poll_seconds: int = 30,
    max_daily_loss_points: float = 170.0,
    trailing_stop: str = "none",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replay EMA signals on index points with a realized-plus-open daily loss cap."""
    if poll_seconds <= 0 or poll_seconds > 120:
        raise ValueError("poll_seconds must be between 1 and 120.")
    if start > end:
        raise ValueError("start date must not be after end date.")
    if not math.isfinite(max_daily_loss_points) or max_daily_loss_points <= 0:
        raise ValueError("max_daily_loss_points must be positive and finite.")
    if trailing_stop not in INDEX_TRAILING_STOPS:
        raise ValueError(f"trailing_stop must be one of {', '.join(INDEX_TRAILING_STOPS)}.")
    if index_bars.empty:
        raise ValueError("Fyers returned no complete NIFTY five-minute bars.")

    enriched = signal_logic.build_ema_trend_with_indicators(index_bars, strategy_config)
    if enriched.empty:
        raise ValueError("The EMA indicator builder returned no usable bars.")
    bar_times = pd.to_datetime(enriched["timestamp"], errors="coerce")
    bar_times = bar_times.dt.tz_localize(IST) if bar_times.dt.tz is None else bar_times.dt.tz_convert(IST)
    indexes_by_day: dict[date, list[int]] = {}
    for index, timestamp in enumerate(bar_times):
        if pd.isna(timestamp):
            continue
        if start <= timestamp.date() <= end:
            indexes_by_day.setdefault(timestamp.date(), []).append(index)

    engine = signal_logic.EMATrendSignalEngine(strategy_config)
    trades: list[dict[str, Any]] = []
    counters = {"signals": 0, "entries": 0, "exits": 0, "daily_loss_halts": 0}
    position: dict[str, Any] | None = None
    daily_realized_points = 0.0
    if trailing_stop != "none":
        required_trailing_columns = {"ema9", "atr"}
        if trailing_stop == "breakeven_1r_ema11":
            required_trailing_columns.add("ema11")
        missing_trailing_columns = required_trailing_columns.difference(enriched.columns)
        if missing_trailing_columns:
            raise ValueError(
                "Trailing stops require indicator columns: "
                + ", ".join(sorted(missing_trailing_columns))
            )

    def close_position(at: pd.Timestamp, spot: float, reason: str) -> None:
        nonlocal position
        nonlocal daily_realized_points
        assert position is not None
        direction_multiplier = 1.0 if position["action"] == "ENTER_LONG" else -1.0
        trade_points = direction_multiplier * (spot - position["entry_price"])
        trades.append(
            {
                "signal_bar_start": position["signal_timestamp"].isoformat(),
                "signal_time": (position["signal_timestamp"] + pd.Timedelta(minutes=5)).isoformat(),
                "entry_time": position["entry_timestamp"].isoformat(),
                "exit_time": at.isoformat(),
                "signal": position["action"],
                "exit_reason": reason,
                "entry_index": position["entry_price"],
                "exit_index": spot,
                "index_points": trade_points,
            }
        )
        counters["exits"] += 1
        daily_realized_points += trade_points
        position = None

    def update_trailing_stop(current_index: int) -> None:
        if position is None or trailing_stop == "none":
            return
        current_bar = enriched.iloc[current_index]
        is_long = position["action"] == "ENTER_LONG"
        if is_long:
            position["high_water"] = max(position["high_water"], float(current_bar["high"]))
        else:
            position["low_water"] = min(position["low_water"], float(current_bar["low"]))

        armed_now = position["trailing_armed"] or (
            position["initial_risk"] > 0
            and (
                position["high_water"] - position["entry_price"] >= position["initial_risk"]
                if is_long
                else position["entry_price"] - position["low_water"] >= position["initial_risk"]
            )
        )
        position["trailing_armed"] = armed_now
        candidates = [position["trailing_stop_price"]]

        if trailing_stop in {"swing_2bar", "combined"} and current_index >= 1:
            swing_bars = enriched.iloc[current_index - 1 : current_index + 1]
            swing_stop = (
                float(swing_bars["low"].min())
                if is_long
                else float(swing_bars["high"].max())
            )
            candidates.append(swing_stop)

        if trailing_stop in {"atr_2x", "combined"}:
            atr = float(current_bar["atr"])
            if math.isfinite(atr) and atr > 0:
                chandelier = (
                    position["high_water"] - 2 * atr
                    if is_long
                    else position["low_water"] + 2 * atr
                )
                candidates.append(chandelier)

        if trailing_stop in {"breakeven_1r_ema9", "breakeven_1r_ema11", "combined"} and armed_now:
            candidates.append(position["entry_price"])
            trailing_ema = "ema11" if trailing_stop == "breakeven_1r_ema11" else "ema9"
            ema_value = float(current_bar[trailing_ema])
            if math.isfinite(ema_value):
                candidates.append(ema_value)

        tightened = max(candidates) if is_long else min(candidates)
        position["trailing_stop_price"] = tightened

    for day in sorted(indexes_by_day):
        day_indexes = indexes_by_day[day]
        daily_realized_points = 0.0
        daily_loss_halted = False
        next_bar = 0
        current_index: int | None = None
        last_processed: pd.Timestamp | None = None
        for now in _action_polls(day, poll_seconds):
            while (
                next_bar < len(day_indexes)
                and bar_times.iloc[day_indexes[next_bar]] + pd.Timedelta(minutes=5) <= now
            ):
                current_index = day_indexes[next_bar]
                next_bar += 1
            current = enriched.iloc[current_index] if current_index is not None else None
            current_timestamp = bar_times.iloc[current_index] if current_index is not None else None
            is_new_bar = current_index is not None and current_timestamp != last_processed

            if (
                position is not None
                and current is not None
                and current_index is not None
                and current_index > position["entry_bar_index"]
            ):
                direction_multiplier = 1.0 if position["action"] == "ENTER_LONG" else -1.0
                trail_price = position["trailing_stop_price"]
                current_close = float(current["close"])
                close_points = direction_multiplier * (current_close - position["entry_price"])
                trail_hit = is_new_bar and trailing_stop != "none" and (
                    current_close <= trail_price
                    if direction_multiplier > 0
                    else current_close >= trail_price
                )
                cap_hit = daily_realized_points + close_points <= -max_daily_loss_points
                if cap_hit:
                    close_position(now, current_close, "DAILY_MAX_LOSS")
                    daily_loss_halted = True
                    counters["daily_loss_halts"] += 1
                    continue
                if trail_hit:
                    close_position(now, current_close, "TRAILING_STOP")
                    if daily_realized_points <= -max_daily_loss_points:
                        daily_loss_halted = True
                        counters["daily_loss_halts"] += 1
                    continue

            if now.timetz().replace(tzinfo=None) >= SQUARE_OFF:
                if position is not None and current is not None:
                    close_position(now, float(current["close"]), "SQUARE_OFF")
                break

            decision = signal_logic.EMATrendDecision()
            if (
                current is not None
                and current_index is not None
                and current_timestamp is not None
                and current_timestamp != last_processed
            ):
                context = None
                if position is not None:
                    context = signal_logic.EMATrendPositionContext(
                        direction="LONG" if position["action"] == "ENTER_LONG" else "SHORT",
                        entry_underlying=position["entry_price"],
                        stop_underlying=float(current.get("ema9", 0.0)),
                    )
                decision = engine.evaluate_candle(enriched.iloc[: current_index + 1], position=context)
                last_processed = current_timestamp
                if decision.signal_triggered:
                    counters["signals"] += 1

            if position is not None and current is not None:
                if decision.action == "EXIT":
                    close_position(now, float(current["close"]), decision.exit_reason or "EMA9_EXIT")
                    if daily_realized_points <= -max_daily_loss_points:
                        daily_loss_halted = True
                        counters["daily_loss_halts"] += 1
                elif is_new_bar and current_index is not None:
                    update_trailing_stop(current_index)
                continue

            if daily_loss_halted or daily_realized_points <= -max_daily_loss_points:
                continue
            if current is None or decision.action not in {"ENTER_LONG", "ENTER_SHORT"}:
                continue
            if current_index is None or current_timestamp is None:
                continue
            close_time = current_timestamp + pd.Timedelta(minutes=5)
            age = now - close_time
            entry_allowed = (
                ENTRY_START <= now.timetz().replace(tzinfo=None) < SQUARE_OFF
                and current_timestamp.date() == day
                and timedelta(0) <= age.to_pytimedelta() <= MAX_ENTRY_BAR_AGE
            )
            if not entry_allowed:
                continue
            entry_price = float(current["close"])
            initial_stop_ema = "ema11" if trailing_stop == "breakeven_1r_ema11" else "ema9"
            entry_stop_price = float(current.get(initial_stop_ema, entry_price))
            position = {
                "action": decision.action,
                "signal_timestamp": current_timestamp,
                "entry_timestamp": now,
                "entry_price": entry_price,
                "entry_bar_index": current_index,
                "trailing_stop_price": entry_stop_price,
                "initial_risk": abs(entry_price - entry_stop_price),
                "high_water": entry_price,
                "low_water": entry_price,
                "trailing_armed": False,
            }
            counters["entries"] += 1

    if position is not None:
        raise RuntimeError("Index replay ended with an open position; the requested data range is incomplete.")
    return pd.DataFrame(trades), counters


def _report_index(
    trades: pd.DataFrame,
    counters: dict[str, int],
    index_name: str,
    start: date,
    end: date,
    poll_seconds: int,
    max_daily_loss_points: float,
    trailing_stop: str,
    exit_ema_period: int,
    history_cache: FyersHistoryCache,
) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trailing_suffix = f"_{trailing_stop}" if trailing_stop != "none" else ""
    exit_ema_suffix = f"_exit_ema{exit_ema_period}" if exit_ema_period != 9 else ""
    cap_suffix = f"_cap{max_daily_loss_points:g}"
    stem = (
        f"fyers_ema_{index_name.lower()}_index_replay_"
        f"{start:%Y%m%d}_{end:%Y%m%d}{trailing_suffix}{exit_ema_suffix}{cap_suffix}"
    )
    trades_path = OUTPUT_DIR / f"{stem}_trades.csv"
    summary_path = OUTPUT_DIR / f"{stem}_summary.txt"
    trades.to_csv(trades_path, index=False)
    points = pd.to_numeric(trades.get("index_points", pd.Series(dtype=float)), errors="coerce").dropna()
    winners, losers = points[points > 0], points[points < 0]
    cumulative = points.cumsum()
    drawdown = cumulative - cumulative.cummax() if not cumulative.empty else pd.Series(dtype=float)
    lines = [
        f"{index_name} index-only EMA replay: {start} through {end}",
        f"Signals use the shared EMA logic on complete Fyers {index_name} five-minute candles.",
        "Entry and exit prices are the index close available at the simulated decision; no option data is used.",
        f"Simulated poll interval: {poll_seconds}s.",
        "Results are directional index points, not rupee P&L; no lot multiplier, brokerage, or slippage is applied.",
        f"Daily loss cap: {max_daily_loss_points:.2f} directional index points, including realized "
        "and close-based open P&L; positions exit at a completed candle close when the cap is breached, "
        "and entries stop for the day.",
        "Daily cap, trailing-stop, and EMA exits are evaluated on completed five-minute candle closes; "
        "the simulated exit price is that candle's close and losses can exceed the cap.",
        f"Additional trailing stop: {trailing_stop}. Stops are checked on later completed bars and "
        f"only tighten. The strategy exit uses a candle-close confirmation on EMA{exit_ema_period}.",
        f"Historical index cache: {'enabled' if history_cache.enabled else 'disabled'}; "
        f"cache hits={history_cache.hits}, history API fetches={history_cache.misses}.",
        f"Signals: {counters['signals']}",
        f"Entries: {counters['entries']}",
        f"Exits: {counters['exits']}",
        f"Daily loss halts: {counters['daily_loss_halts']}",
        f"Completed trades: {len(trades)}",
        f"Win rate: {(len(winners) / len(points) * 100.0) if len(points) else 0.0:.2f}%",
        f"Average winning points: {float(winners.mean()) if not winners.empty else 0.0:.2f}",
        f"Average losing points: {float(losers.mean()) if not losers.empty else 0.0:.2f}",
        f"Net directional index points: {float(points.sum()) if not points.empty else 0.0:.2f}",
        f"Max trade-equity drawdown (points): {float(drawdown.min()) if not drawdown.empty else 0.0:.2f}",
        f"Trades CSV: {trades_path}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return trades_path, summary_path


def replay(
    *,
    signal_logic: Any,
    strategy_config: Any,
    index_bars: pd.DataFrame,
    api: FyersAPI,
    active_rows: list[dict[str, Any]],
    expiries: list[date],
    start: date,
    end: date,
    lots: int = 1,
    max_daily_loss: float = 5500.0,
    poll_seconds: int = 30,
    option_resolution: str = "5S",
    history_cache: FyersHistoryCache | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if poll_seconds <= 0 or poll_seconds > 120:
        raise ValueError("poll_seconds must be between 1 and 120.")
    if start > end:
        raise ValueError("start date must not be after end date.")
    if lots <= 0 or not math.isfinite(max_daily_loss) or max_daily_loss <= 0:
        raise ValueError("lots and max_daily_loss must be positive and finite.")
    if option_resolution not in OPTION_RESOLUTION_SECONDS:
        raise ValueError(f"Unsupported option resolution: {option_resolution}")
    if index_bars.empty:
        raise ValueError("Fyers returned no complete NIFTY five-minute bars.")

    enriched = signal_logic.build_ema_trend_with_indicators(index_bars, strategy_config)
    if enriched.empty:
        raise ValueError("The EMA indicator builder returned no usable bars.")
    bar_times = pd.to_datetime(enriched["timestamp"], errors="coerce")
    bar_times = bar_times.dt.tz_localize(IST) if bar_times.dt.tz is None else bar_times.dt.tz_convert(IST)
    indexes_by_day: dict[date, list[int]] = {}
    for index, timestamp in enumerate(bar_times):
        if pd.isna(timestamp):
            continue
        day = timestamp.date()
        if start <= day <= end:
            indexes_by_day.setdefault(day, []).append(index)

    engine = signal_logic.EMATrendSignalEngine(strategy_config)
    eligible_rows = [
        row for row in active_rows if row.get("underlying") == "NIFTY" and row.get("option_type") in {"CE", "PE"}
    ]
    if not eligible_rows:
        raise RuntimeError("Fyers derivatives master has no NIFTY option contracts.")
    fallback_lot_size = int(eligible_rows[0]["lot_size"])
    archive_cache: dict[date, list[str]] = {}
    option_cache: dict[tuple[str, date], _OptionQuotes] = {}
    trades: list[dict[str, Any]] = []
    counters = {"signals": 0, "entries": 0, "exits": 0, "daily_loss_halts": 0}
    position: _Position | None = None

    def option_day(contract: FyersOptionContract, day: date) -> _OptionQuotes:
        key = (contract.symbol, day)
        if key not in option_cache:
            expired = contract.expiry < date.today()
            history = (
                history_cache.history(
                    api,
                    contract.symbol,
                    resolution=option_resolution,
                    start=day,
                    end=day,
                    expired=expired,
                )
                if history_cache is not None
                else api.history(
                    contract.symbol,
                    resolution=option_resolution,
                    start=day,
                    end=day,
                    expired=expired,
                )
            )
            option_cache[key] = _prepare_option_quotes(history, option_resolution)
        return option_cache[key]

    def close_trade(
        *,
        at: pd.Timestamp,
        spot: float,
        price: float,
        reason: str,
        quote_time: pd.Timestamp,
    ) -> None:
        nonlocal position
        assert position is not None
        pnl = (price - position.entry_option) * position.quantity
        trades.append(
            {
                "signal_time": (position.signal_timestamp + pd.Timedelta(minutes=5)).isoformat(),
                "signal_bar_start": position.signal_timestamp.isoformat(),
                "entry_time": position.entry_timestamp.isoformat(),
                "entry_quote_time": position.entry_quote_timestamp.isoformat(),
                "exit_time": at.isoformat(),
                "exit_quote_time": quote_time.isoformat(),
                "signal": position.action,
                "exit_reason": reason,
                "symbol": position.contract.symbol,
                "expiry": position.contract.expiry.isoformat(),
                "option_type": position.contract.option_type,
                "strike": position.contract.strike,
                "lot_size": position.contract.lot_size,
                "quantity": position.quantity,
                "entry_spot": position.entry_spot,
                "exit_spot": spot,
                "entry_option": position.entry_option,
                "exit_option": price,
                "gross_pnl": pnl,
                "option_resolution": option_resolution,
            }
        )
        counters["exits"] += 1
        position = None

    for day in sorted(indexes_by_day):
        daily_realized = 0.0
        last_processed: pd.Timestamp | None = None
        day_indexes = indexes_by_day[day]
        next_bar = 0
        current_index: int | None = None
        for now in _action_polls(day, poll_seconds):
            while (
                next_bar < len(day_indexes) and bar_times.iloc[day_indexes[next_bar]] + pd.Timedelta(minutes=5) <= now
            ):
                current_index = day_indexes[next_bar]
                next_bar += 1
            current = enriched.iloc[current_index] if current_index is not None else None
            current_timestamp = bar_times.iloc[current_index] if current_index is not None else None

            if now.timetz().replace(tzinfo=None) >= SQUARE_OFF:
                if position is not None:
                    quote = _quote_at(position.option_quotes, now, option_resolution)
                    if quote is None:
                        raise RuntimeError(
                            f"Missing historical option quote at square-off for {position.contract.symbol} on {day}."
                        )
                    spot = float(current["close"]) if current is not None else position.entry_spot
                    close_trade(
                        at=now,
                        spot=spot,
                        price=quote[0],
                        reason="SQUARE_OFF",
                        quote_time=quote[1],
                    )
                    daily_realized += trades[-1]["gross_pnl"]
                break

            decision = signal_logic.EMATrendDecision()
            if current is not None and current_timestamp != last_processed:
                context = None
                if position is not None:
                    context = signal_logic.EMATrendPositionContext(
                        direction=position.direction,
                        entry_underlying=position.entry_spot,
                        stop_underlying=float(current.get("ema9", 0.0)),
                    )
                prefix = enriched.iloc[: current_index + 1]
                decision = engine.evaluate_candle(prefix, position=context)
                last_processed = current_timestamp
                if decision.signal_triggered:
                    counters["signals"] += 1

            if position is not None:
                quote = _quote_at(position.option_quotes, now, option_resolution)
                if quote is None:
                    raise RuntimeError(
                        f"Missing historical option quote at {now.isoformat()} for {position.contract.symbol}."
                    )
                risk_exit = daily_realized + (quote[0] - position.entry_option) * position.quantity <= -max_daily_loss
                if risk_exit or (decision.action == "EXIT" and decision.exit_reason):
                    reason = "DAILY_MAX_LOSS" if risk_exit else decision.exit_reason
                    spot = float(current["close"]) if current is not None else position.entry_spot
                    exit_at = now
                    exit_quote = quote if risk_exit else _quote_at(position.option_quotes, exit_at, option_resolution)
                    if exit_quote is None:
                        raise RuntimeError(
                            f"Missing historical option quote for {reason} at {exit_at.isoformat()} "
                            f"for {position.contract.symbol}."
                        )
                    close_trade(
                        at=exit_at,
                        spot=spot,
                        price=exit_quote[0],
                        reason=reason,
                        quote_time=exit_quote[1],
                    )
                    daily_realized += trades[-1]["gross_pnl"]
                    if risk_exit:
                        counters["daily_loss_halts"] += 1
                    continue
                continue

            if current is None or decision.action not in {"ENTER_LONG", "ENTER_SHORT"}:
                continue
            close_time = current_timestamp + pd.Timedelta(minutes=5)
            age = now - close_time
            entry_allowed = (
                ENTRY_START <= now.timetz().replace(tzinfo=None) < SQUARE_OFF
                and current_timestamp.date() == day
                and timedelta(0) <= age.to_pytimedelta() <= MAX_ENTRY_BAR_AGE
                and daily_realized > -max_daily_loss
            )
            if not entry_allowed:
                continue
            contract = _resolve_contract(
                api,
                action=decision.action,
                spot=float(current["close"]),
                trade_date=day,
                active_rows=active_rows,
                expiries=expiries,
                archived_symbols=archive_cache,
                fallback_lot_size=fallback_lot_size,
            )
            quotes = option_day(contract, day)
            entry_at = now
            quote = _quote_at(quotes, entry_at, option_resolution)
            if quote is None:
                raise RuntimeError(
                    f"Missing historical option quote at entry {entry_at.isoformat()} for {contract.symbol}."
                )
            position = _Position(
                contract=contract,
                action=decision.action,
                direction="LONG" if decision.action == "ENTER_LONG" else "SHORT",
                quantity=contract.lot_size * lots,
                signal_timestamp=current_timestamp,
                entry_timestamp=entry_at,
                entry_quote_timestamp=quote[1],
                entry_spot=float(current["close"]),
                entry_option=quote[0],
                option_quotes=quotes,
            )
            counters["entries"] += 1

    if position is not None:
        raise RuntimeError("Replay ended with an open position; the requested data range is incomplete.")
    return pd.DataFrame(trades), counters


def _report(
    trades: pd.DataFrame,
    counters: dict[str, int],
    start: date,
    end: date,
    option_resolution: str,
    poll_seconds: int,
    history_cache: FyersHistoryCache,
) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"fyers_ema_replay_{start:%Y%m%d}_{end:%Y%m%d}"
    trades_path = OUTPUT_DIR / f"{stem}_trades.csv"
    summary_path = OUTPUT_DIR / f"{stem}_summary.txt"
    trades.to_csv(trades_path, index=False)
    pnl = pd.to_numeric(trades.get("gross_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    winners, losers = pnl[pnl > 0], pnl[pnl < 0]
    profit_factor = float(winners.sum() / abs(losers.sum())) if not losers.empty else float("inf")
    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax() if not cumulative.empty else pd.Series(dtype=float)
    if trades.empty:
        daily_pnl: dict[str, float] = {}
    else:
        entry_days = pd.to_datetime(trades["entry_time"], errors="coerce").dt.date.astype(str)
        daily_pnl = trades.assign(entry_day=entry_days).groupby("entry_day")["gross_pnl"].sum().to_dict()
    lines = [
        f"Live-process Fyers EMA replay: {start} through {end}",
        "Signals use the shared EMA logic on complete Fyers NIFTY five-minute bars.",
        "Entry and EMA9 exit decisions use the option quote proxy at the same simulated poll; "
        "daily loss is checked on each poll. Quotes use the last completed option candle and are "
        "not guaranteed live fills.",
        f"Option candle resolution: {option_resolution}; simulated poll interval: {poll_seconds}s.",
        f"Historical candle cache: {'enabled' if history_cache.enabled else 'disabled'}; "
        f"cache hits={history_cache.hits}, history API fetches={history_cache.misses}.",
        "Entry window, EMA9 exit, daily max-loss, and 15:15 square-off rules mirror the live runner.",
        "P&L is gross, matching the live alert calculation; fees and slippage are not included.",
        "Archived contracts use the current Fyers lot size because historical lot-size metadata "
        "is not supplied by the expired-symbol endpoint.",
        f"Signals: {counters['signals']}",
        f"Entries: {counters['entries']}",
        f"Exits: {counters['exits']}",
        f"Completed trades: {len(trades)}",
        f"Daily max-loss exits: {counters['daily_loss_halts']}",
        f"Win rate: {(len(winners) / len(pnl) * 100.0) if len(pnl) else 0.0:.2f}%",
        f"Average win: {float(winners.mean()) if not winners.empty else 0.0:.2f}",
        f"Average loss: {float(losers.mean()) if not losers.empty else 0.0:.2f}",
        f"Gross option P&L: {float(pnl.sum()):.2f}",
        f"Profit factor: {profit_factor:.4f}",
        f"Max trade-equity drawdown: {float(drawdown.min()) if not drawdown.empty else 0.0:.2f}",
        "Daily gross P&L: " + (", ".join(f"{day}={value:.2f}" for day, value in daily_pnl.items()) or "none"),
        f"Trades CSV: {trades_path}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return trades_path, summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay EMA signals using Fyers index or option history.")
    parser.add_argument("--start-date", required=True, help="Inclusive YYYY-MM-DD date.")
    parser.add_argument("--end-date", required=True, help="Inclusive YYYY-MM-DD date.")
    parser.add_argument(
        "--instrument",
        choices=("options", "index"),
        default="options",
        help="Replay using option-premium estimates (default) or directional index points.",
    )
    parser.add_argument(
        "--index",
        choices=tuple(INDEX_SYMBOLS),
        default="nifty",
        help="Underlying index for --instrument index (default: nifty). Options replay supports NIFTY only.",
    )
    parser.add_argument("--lots", type=int, default=None)
    parser.add_argument("--max-daily-loss", type=float, default=None)
    parser.add_argument(
        "--max-daily-loss-points",
        type=float,
        default=170.0,
        help="Index-only daily realized-plus-open loss cap in directional index points (default: 170).",
    )
    parser.add_argument("--poll-seconds", type=int, default=None)
    parser.add_argument(
        "--trailing-stop",
        choices=INDEX_TRAILING_STOPS,
        default=None,
        help=(
            "Additional stop for index-only replay: none, swing_2bar, atr_2x, "
            "breakeven_1r_ema9, breakeven_1r_ema11, or combined "
            "(index default: breakeven_1r_ema9)."
        ),
    )
    parser.add_argument(
        "--exit-ema",
        choices=(9, 11),
        type=int,
        default=9,
        help="EMA period for the normal strategy exit in index-only replay (default: 9).",
    )
    parser.add_argument("--option-resolution", choices=("auto", "5S", "1"), default="auto")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--useCache",
        "--use-cache",
        dest="use_cache",
        type=_parse_cache_flag,
        default=None,
        metavar="{true,false}",
        help="Read/write historical candle cache (default: EMA_REPLAY_USE_CACHE or true).",
    )
    cache_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Fetch all historical candles directly from Fyers; do not read or write the local cache.",
    )
    cache_group.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Fetch candles from Fyers and replace matching cache entries.",
    )
    args = parser.parse_args(argv)
    trailing_stop = args.trailing_stop
    if trailing_stop is None:
        trailing_stop = "breakeven_1r_ema9" if args.instrument == "index" else "none"
    try:
        start, end = date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)
    except ValueError as exc:
        parser.error(str(exc))
    if start > end or (args.lots is not None and args.lots <= 0):
        parser.error("date range must be ordered and lots must be positive")
    if not math.isfinite(args.max_daily_loss_points) or args.max_daily_loss_points <= 0:
        parser.error("--max-daily-loss-points must be positive and finite.")

    load_dotenv(ENV_FILE, override=False)
    if args.use_cache is None:
        try:
            use_cache = _parse_cache_flag(os.getenv("EMA_REPLAY_USE_CACHE", "true"))
        except argparse.ArgumentTypeError as exc:
            parser.error(f"Invalid EMA_REPLAY_USE_CACHE: {exc}")
    else:
        use_cache = args.use_cache
    if args.no_cache:
        use_cache = False
    app_id = os.getenv("FYERS_APP_ID", "").strip()
    token_file_value = os.getenv("FYERS_TOKEN_FILE", "").strip()
    token_file = (
        Path(token_file_value).expanduser() if token_file_value else ROOT_DIR / "Dependencies" / ".fyers_access_token"
    )
    if not token_file.is_absolute():
        token_file = ROOT_DIR / token_file
    access_token = load_persisted_access_token(token_file, os.getenv("FYERS_ACCESS_TOKEN", ""))
    if not app_id or not access_token:
        parser.error("Set FYERS_APP_ID and a valid FYERS_ACCESS_TOKEN/token file in Dependencies/.env")

    try:
        poll_seconds = args.poll_seconds if args.poll_seconds is not None else int(os.getenv("EMA_POLL_SECONDS", "30"))
        if args.instrument == "options":
            lots = args.lots if args.lots is not None else int(os.getenv("EMA_LOTS", "1"))
            max_daily_loss = (
                args.max_daily_loss if args.max_daily_loss is not None else float(os.getenv("EMA_MAX_LOSS", "5500"))
            )
    except ValueError as exc:
        parser.error(f"Invalid EMA replay configuration: {exc}")
    if poll_seconds <= 0 or poll_seconds > 120:
        parser.error("poll-seconds must be between 1 and 120.")
    if args.instrument == "options":
        if args.exit_ema != 9:
            parser.error("--exit-ema 11 is supported only with --instrument index.")
        if args.index != "nifty":
            parser.error("--index sensex is only supported with --instrument index.")
        if lots <= 0 or not math.isfinite(max_daily_loss) or max_daily_loss <= 0:
            parser.error("lots and max-daily-loss must be positive; loss must be finite.")
        if os.getenv("EMA_OPTION_EXPIRY_POLICY", "NEXT_WEEK").strip().upper() != "NEXT_WEEK":
            parser.error("This replay supports EMA_OPTION_EXPIRY_POLICY=NEXT_WEEK only.")
        if os.getenv("EMA_OPTION_STRIKE_POLICY", "ATM").strip().upper() != "ATM":
            parser.error("This replay supports EMA_OPTION_STRIKE_POLICY=ATM only.")
    option_resolution = args.option_resolution
    if option_resolution == "auto":
        option_resolution = "5S" if start >= date.today() - timedelta(days=35) else "1"
    if start > date.today():
        parser.error("start-date cannot be in the future")
    now = datetime.now(IST)
    if end > now.date() or (end == now.date() and now.timetz().replace(tzinfo=None) < SQUARE_OFF):
        parser.error("end-date must be a completed trading date; today's replay is available after 15:15 IST.")
    query_end = _expiry_lookup_end(end, date.today())
    logic = _load_signal_logic()
    config = logic.EMATrendConfig(
        ema_fast_period=4,
        ema_mid_period=11,
        ema_slow_period=18,
        atr_period=14,
        adx_period=14,
        slope_lookback=3,
        adx_threshold=20.0,
        distance_atr_multiplier=0.5,
        ema11_slope_atr_multiplier=0.3,
        ema18_slope_atr_multiplier=0.2,
        exit_ema_period=args.exit_ema,
    )
    api = FyersAPI(app_id, access_token)
    history_cache = FyersHistoryCache(
        OUTPUT_DIR / ".fyers_cache",
        enabled=use_cache,
        refresh=args.refresh_cache,
    )
    try:
        index_name, index_symbol = INDEX_SYMBOLS[args.index]
        raw_index = _fetch_index_history(
            api,
            _index_history_warmup_start(start, date.today()),
            end,
            history_cache,
            symbol=index_symbol,
        )
        if raw_index.empty:
            raise RuntimeError(f"Fyers returned no {index_name} index candles for the selected period.")
        end_of_range = datetime.combine(end + timedelta(days=1), time.min, tzinfo=IST)
        index_bars = completed_five_minute_bars(raw_index, end_of_range)
        if args.instrument == "index":
            trades, counters = replay_index_only(
                signal_logic=logic,
                strategy_config=config,
                index_bars=index_bars,
                index_name=index_name,
                start=start,
                end=end,
                poll_seconds=poll_seconds,
                max_daily_loss_points=args.max_daily_loss_points,
                trailing_stop=trailing_stop,
            )
            _report_index(
                trades,
                counters,
                index_name,
                start,
                end,
                poll_seconds,
                args.max_daily_loss_points,
                trailing_stop,
                args.exit_ema,
                history_cache,
            )
            return 0

        active_rows = api.option_master()
        active_expiries = {
            expiry
            for row in active_rows
            if row.get("underlying") == "NIFTY" and (expiry := _active_expiry(row)) is not None
        }
        archived_expiries: list[date] = []
        if start <= query_end:
            for left, right in _history_chunks(start, query_end):
                archived_expiries.extend(api.expired_option_expiries(UNDERLYING_SYMBOL, left, right))
        expiries = sorted(active_expiries | set(archived_expiries))
        trades, counters = replay(
            signal_logic=logic,
            strategy_config=config,
            index_bars=index_bars,
            api=api,
            active_rows=active_rows,
            expiries=expiries,
            start=start,
            end=end,
            lots=lots,
            max_daily_loss=max_daily_loss,
            poll_seconds=poll_seconds,
            option_resolution=option_resolution,
            history_cache=history_cache,
        )
    except (FyersAPIError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    _report(trades, counters, start, end, option_resolution, poll_seconds, history_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
