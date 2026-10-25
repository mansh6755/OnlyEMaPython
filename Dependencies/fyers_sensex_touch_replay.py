"""Run paired SENSEX EMA9-touch index replays with configurable daily loss caps."""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Dependencies.fyers_api import FyersAPI  # noqa: E402
from Dependencies.fyers_ema_replay import (  # noqa: E402
    ENV_FILE,
    IST,
    FyersHistoryCache,
    _history_chunks,
    _index_history_warmup_start,
    _load_signal_logic,
)
from Dependencies.fyers_market_data import completed_five_minute_bars  # noqa: E402
from Dependencies.fyers_telegram_auth import load_persisted_access_token  # noqa: E402

INDEX_SYMBOL = "BSE:SENSEX-INDEX"
DEFAULT_START = date(2022, 1, 1)
DEFAULT_END = date(2026, 10, 23)
DEFAULT_CAPS = (250.0, 300.0)
ENTRY_START = time(9, 25)
SQUARE_OFF = time(15, 15)
BASELINE_TRADES = (
    ROOT_DIR
    / "Backtest Outputs"
    / "fyers_ema_sensex_index_replay_20260101_20261022_ema9_touch_1m_proxy_cap250_trades.csv"
)


def fetch_minute_history(
    start: date,
    end: date,
    *,
    cache_enabled: bool = True,
    refresh_cache: bool = False,
) -> tuple[pd.DataFrame, FyersHistoryCache]:
    """Fetch SENSEX history in bounded requests, optionally using the shared cache."""
    load_dotenv(ENV_FILE, override=False)
    app_id = os.getenv("FYERS_APP_ID", "").strip()
    token_file_value = os.getenv("FYERS_TOKEN_FILE", "").strip()
    token_file = (
        Path(token_file_value).expanduser() if token_file_value else ROOT_DIR / "Dependencies" / ".fyers_access_token"
    )
    if not token_file.is_absolute():
        token_file = ROOT_DIR / token_file
    access_token = load_persisted_access_token(token_file, os.getenv("FYERS_ACCESS_TOKEN", ""))
    if not app_id or not access_token:
        raise RuntimeError("Set FYERS_APP_ID and a valid FYERS_ACCESS_TOKEN/token file in Dependencies/.env.")

    api = FyersAPI(app_id, access_token)
    api.validate_session()
    cache = FyersHistoryCache(
        ROOT_DIR / "Backtest Outputs" / ".fyers_cache",
        enabled=cache_enabled,
        refresh=refresh_cache,
    )
    request_start = _index_history_warmup_start(start, date.today())
    chunks: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _history_chunks(request_start, end):
        chunks.append(
            cache.history(
                api,
                INDEX_SYMBOL,
                resolution="1",
                start=chunk_start,
                end=chunk_end,
            )
        )
    non_empty = [chunk for chunk in chunks if not chunk.empty]
    if not non_empty:
        raise RuntimeError("Fyers returned no SENSEX minute candles for the requested period.")
    minute_bars = (
        pd.concat(non_empty, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    return minute_bars, cache


def replay_sensex_touch_cap(
    enriched_bars: pd.DataFrame,
    minute_bars: pd.DataFrame,
    *,
    start: date,
    end: date,
    daily_loss_cap: float,
    ema9_buffer_points: float = 0.0,
    signal_logic: Any,
    strategy_config: Any,
) -> pd.DataFrame:
    """Replay EMA9 touch exits with five-minute-close trail and daily-cap checks."""
    if not math.isfinite(daily_loss_cap) or daily_loss_cap <= 0:
        raise ValueError("daily_loss_cap must be positive and finite.")
    if not math.isfinite(ema9_buffer_points) or ema9_buffer_points < 0:
        raise ValueError("ema9_buffer_points must be finite and non-negative.")
    if enriched_bars.empty:
        raise ValueError("No completed SENSEX five-minute bars are available.")

    bar_times = pd.to_datetime(enriched_bars["timestamp"], errors="coerce")
    bar_times = bar_times.dt.tz_localize(IST) if bar_times.dt.tz is None else bar_times.dt.tz_convert(IST)
    if bar_times.isna().any():
        raise ValueError("SENSEX five-minute bars contain invalid timestamps.")

    minutes = minute_bars.copy()
    if "timestamp" not in minutes:
        raise ValueError("SENSEX minute data is missing its timestamp column.")
    minutes["timestamp"] = pd.to_datetime(minutes["timestamp"], errors="coerce")
    if minutes["timestamp"].dt.tz is None:
        minutes["timestamp"] = minutes["timestamp"].dt.tz_localize(IST)
    else:
        minutes["timestamp"] = minutes["timestamp"].dt.tz_convert(IST)
    minutes = minutes.dropna(subset=["timestamp"])
    minutes = minutes[minutes["timestamp"].dt.date.between(start, end)].copy()
    minutes["bar_start"] = minutes["timestamp"].dt.floor("5min")
    minutes_by_bar = {key: group for key, group in minutes.groupby("bar_start", sort=True)}  # noqa: C416

    trades: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None
    daily_realized_points = 0.0
    daily_loss_halted = False
    current_day: date | None = None
    engine = signal_logic.EMATrendSignalEngine(strategy_config)

    def close_position(at: pd.Timestamp, price: float, reason: str) -> None:
        nonlocal position, daily_realized_points, daily_loss_halted
        assert position is not None
        direction = 1.0 if position["signal"] == "ENTER_LONG" else -1.0
        points = direction * (price - position["entry_price"])
        trades.append(
            {
                "signal_bar_start": position["signal_bar_start"].isoformat(),
                "entry_time": position["entry_time"].isoformat(),
                "exit_time": at.isoformat(),
                "signal": position["signal"],
                "exit_reason": reason,
                "entry_index": position["entry_price"],
                "exit_index": price,
                "index_points": points,
            }
        )
        daily_realized_points += points

        # Centralize cap check so all exits correctly trigger the daily halt
        if daily_realized_points <= -daily_loss_cap:
            daily_loss_halted = True

        position = None

    for bar_index, bar in enriched_bars.iterrows():
        bar_start = bar_times.iloc[bar_index]
        trading_day = bar_start.date()
        if not start <= trading_day <= end:
            continue
        if trading_day != current_day:
            current_day = trading_day
            daily_realized_points = 0.0
            daily_loss_halted = False

        bar_close_time = bar_start + pd.Timedelta(minutes=5)
        position_open_at_bar_start = position is not None
        minute_group = minutes_by_bar.get(bar_start)

        if position is not None and bar_index > position["entry_bar_index"]:
            prior_ema9 = float(enriched_bars.iloc[bar_index - 1]["ema9"]) if bar_index > 0 else np.nan
            direction = 1.0 if position["signal"] == "ENTER_LONG" else -1.0
            prior_stop = prior_ema9 - direction * ema9_buffer_points

            # 1. Intra-bar checks (EMA9 Touch via minute data)
            if minute_group is not None and math.isfinite(prior_stop):
                for minute in minute_group.itertuples(index=False):
                    if position is None:
                        break
                    if minute.timestamp < position["entry_time"]:
                        continue

                    minute_close_time = minute.timestamp + pd.Timedelta(minutes=1)
                    if position["signal"] == "ENTER_LONG" and float(minute.low) <= prior_stop:
                        fill = float(minute.open) if float(minute.open) <= prior_stop else prior_stop
                        close_position(minute_close_time, fill, "EMA9_TOUCH")
                    elif position["signal"] == "ENTER_SHORT" and float(minute.high) >= prior_stop:
                        fill = float(minute.open) if float(minute.open) >= prior_stop else prior_stop
                        close_position(minute_close_time, fill, "EMA9_TOUCH")

            # 2. End-of-bar checks (Trail, Cap, Square-off) evaluated sequentially at the 5-min close
            if position is not None:
                direction = 1.0 if position["signal"] == "ENTER_LONG" else -1.0
                close_points = direction * (float(bar["close"]) - position["entry_price"])

                if daily_realized_points + close_points <= -daily_loss_cap:
                    close_position(bar_close_time, float(bar["close"]), "DAILY_MAX_LOSS")
                else:
                    trailing_stop = position["trailing_stop"]
                    trail_hit = (
                        float(bar["close"]) <= trailing_stop if direction > 0 else float(bar["close"]) >= trailing_stop
                    )
                    if trail_hit:
                        close_position(bar_close_time, float(bar["close"]), "TRAILING_STOP")
                    elif bar_close_time.time() >= SQUARE_OFF:
                        close_position(bar_close_time, float(bar["close"]), "SQUARE_OFF")
                    else:
                        # Ratchet trailing stop for the next candle if position survives
                        current_ema9 = float(bar["ema9"])
                        if math.isfinite(current_ema9):
                            buffered_ema9 = current_ema9 - direction * ema9_buffer_points
                            position["trailing_stop"] = (
                                max(trailing_stop, buffered_ema9)
                                if direction > 0
                                else min(trailing_stop, buffered_ema9)
                            )

        if position is not None or position_open_at_bar_start:
            continue
        if (
            bar_close_time.time() >= SQUARE_OFF
            or daily_loss_halted
            or daily_realized_points <= -daily_loss_cap
            or bar_close_time.time() < ENTRY_START
        ):
            continue

        decision = engine.evaluate_candle(enriched_bars.iloc[: bar_index + 1])
        if decision.action not in {"ENTER_LONG", "ENTER_SHORT"}:
            continue
        entry_ema9 = float(bar["ema9"])
        if not math.isfinite(entry_ema9):
            raise RuntimeError(f"Missing SENSEX EMA9 at entry time {bar_start}.")
        direction = 1.0 if decision.action == "ENTER_LONG" else -1.0
        position = {
            "signal_bar_start": bar_start,
            "entry_time": bar_close_time,
            "signal": decision.action,
            "entry_price": float(bar["close"]),
            "entry_bar_index": bar_index,
            "trailing_stop": entry_ema9 - direction * ema9_buffer_points,
        }

    if position is not None:
        raise RuntimeError("SENSEX replay ended with an open position.")
    return pd.DataFrame(trades)


def _trade_metrics(trades: pd.DataFrame, cap: float, ema9_buffer_points: float = 0.0) -> dict[str, Any]:
    points = pd.to_numeric(trades["index_points"], errors="raise")
    equity = points.cumsum()
    # Fixed drawdown calculation to yield a positive magnitude
    drawdown = equity.cummax() - equity
    cap_exit_days = trades.loc[trades["exit_reason"] == "DAILY_MAX_LOSS", "exit_time"].str[:10].nunique()
    return {
        "trades": len(trades),
        "wins": int((points > 0).sum()),
        "losses": int((points < 0).sum()),
        "win_rate": float((points > 0).mean() * 100.0) if len(points) else 0.0,
        "net_points": float(points.sum()),
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "cap_hit_days": int(cap_exit_days),
        "exit_counts": trades["exit_reason"].value_counts().to_dict(),
        "cap": cap,
        "ema9_buffer_points": ema9_buffer_points,
    }


def _write_report(
    trades: pd.DataFrame,
    *,
    start: date,
    end: date,
    metrics: dict[str, Any],
    minute_rows: int,
    five_minute_bars: int,
    cache: FyersHistoryCache,
) -> tuple[Path, Path]:
    output_dir = ROOT_DIR / "Backtest Outputs"
    stem = f"fyers_ema_sensex_{start:%Y%m%d}_{end:%Y%m%d}_touch_cap{metrics['cap']:g}_5m_close"
    if metrics["ema9_buffer_points"] > 0:
        stem += f"_ema9buffer{metrics['ema9_buffer_points']:g}"
    trades_path = output_dir / f"{stem}_trades.csv"
    summary_path = output_dir / f"{stem}_summary.txt"
    trades.to_csv(trades_path, index=False)
    monthly = trades.assign(exit_month=pd.to_datetime(trades["exit_time"]).dt.strftime("%Y-%m"))
    monthly_points = monthly.groupby("exit_month")["index_points"].sum()
    lines = [
        f"SENSEX EMA9-touch replay, {start} through {end}",
        (
            "Exploratory reconstructed replay; compare the 250-point result against the saved baseline "
            "before treating the cap comparison as verified."
        ),
        (
            f"EMA9 touch: one-minute OHLC against the prior completed five-minute EMA9 with a "
            f"{metrics['ema9_buffer_points']:g}-point directional buffer; threshold fill adjusted to "
            "minute open if gapped through."
        ),
        (
            f"EMA9 trailing stop starts at entry signal-bar EMA9 with a "
            f"{metrics['ema9_buffer_points']:g}-point directional buffer; existing stop is checked on "
            "completed five-minute close, then buffered EMA9 is ratcheted for the next close."
        ),
        (
            f"Daily cap: realized plus close-based open points checked at each completed five-minute "
            f"close; exit at that close at or below -{metrics['cap']:g} points and halt entries for the day."
        ),
        (
            "Entry at completed five-minute signal close from 09:25 IST; square-off at 15:15 IST. "
            "No 1R breakeven, fees, spread, slippage, latency, or market impact."
        ),
        (
            f"Data: minute rows={minute_rows}; complete five-minute bars={five_minute_bars}; "
            f"cache hits={cache.hits}; API fetches={cache.misses}."
        ),
        (
            f"Trades={metrics['trades']}; wins={metrics['wins']}; losses={metrics['losses']}; "
            f"win rate={metrics['win_rate']:.2f}%."
        ),
        (
            f"Net directional points={metrics['net_points']:.2f}; "
            f"max trade-equity drawdown={metrics['max_drawdown']:.2f}; "
            f"cap-hit days={metrics['cap_hit_days']}."
        ),
        f"Exit counts={metrics['exit_counts']}.",
        "Monthly directional points grouped by exit month:",
        *[f"{month}: {value:+.2f}" for month, value in monthly_points.items()],
        f"Trades CSV: {trades_path}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return trades_path, summary_path


def _compare_saved_baseline(trades: pd.DataFrame) -> str:
    if not BASELINE_TRADES.is_file():
        return "Saved 250-point baseline CSV is unavailable; exact baseline validation was not possible."
    saved = pd.read_csv(BASELINE_TRADES)
    columns = [
        "signal_bar_start",
        "entry_time",
        "exit_time",
        "signal",
        "exit_reason",
        "entry_index",
        "exit_index",
        "index_points",
    ]
    fresh = trades[columns].copy()
    expected = saved[columns].copy()
    for column in ("entry_index", "exit_index", "index_points"):
        fresh[column] = fresh[column].round(8)
        expected[column] = expected[column].round(8)
    merged = fresh.merge(
        expected,
        on=["signal_bar_start", "entry_time", "signal"],
        how="outer",
        suffixes=("_fresh", "_saved"),
        indicator=True,
    )
    same = (
        (merged["_merge"] == "both")
        & (merged["exit_time_fresh"] == merged["exit_time_saved"])
        & (merged["exit_reason_fresh"] == merged["exit_reason_saved"])
        & (merged["exit_index_fresh"] == merged["exit_index_saved"])
        & (merged["index_points_fresh"] == merged["index_points_saved"])
    )
    return (
        f"Saved baseline check: saved={len(saved)} trades/{saved['index_points'].sum():.2f} points; "
        f"fresh={len(trades)} trades/{trades['index_points'].sum():.2f} points; "
        f"exact matching trades={int(same.sum())}/{len(saved)}."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=DEFAULT_START,
        help=f"First trading date to replay (default: {DEFAULT_START}).",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=DEFAULT_END,
        help=f"Last trading date to replay (default: {DEFAULT_END}).",
    )
    parser.add_argument(
        "--caps",
        type=float,
        nargs="+",
        default=list(DEFAULT_CAPS),
        help="One or more daily loss caps in index points (default: 250 300).",
    )
    parser.add_argument(
        "--ema9-buffer-points",
        type=float,
        default=3.0,
        help="Move LONG EMA9 stops down and SHORT EMA9 stops up by this many points (default: 3).",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--refresh-cache", action="store_true", help="Fetch again and replace cached requests.")
    cache_group.add_argument("--no-cache", action="store_true", help="Fetch without reading or writing local cache.")
    args = parser.parse_args(argv)
    if args.start_date > args.end_date:
        parser.error("--start-date must not be after --end-date.")
    if not args.caps or any(not math.isfinite(cap) or cap <= 0 for cap in args.caps):
        parser.error("--caps values must be positive finite point amounts.")
    if len(set(args.caps)) != len(args.caps):
        parser.error("--caps must not contain duplicates.")
    if not math.isfinite(args.ema9_buffer_points) or args.ema9_buffer_points < 0:
        parser.error("--ema9-buffer-points must be finite and non-negative.")

    minute_history, cache = fetch_minute_history(
        args.start_date,
        args.end_date,
        cache_enabled=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )

    # Ensure minute timestamps are localized to IST before filtering via dt.date
    if "timestamp" in minute_history:
        if minute_history["timestamp"].dt.tz is None:
            minute_history["timestamp"] = minute_history["timestamp"].dt.tz_localize(IST)
        else:
            minute_history["timestamp"] = minute_history["timestamp"].dt.tz_convert(IST)

    five_minute = completed_five_minute_bars(
        minute_history,
        datetime.combine(args.end_date + timedelta(days=1), time.min, tzinfo=IST),
    )
    if five_minute.empty:
        raise RuntimeError("No complete SENSEX five-minute bars were built from Fyers minute history.")

    logic = _load_signal_logic()
    config = logic.EMATrendConfig(exit_ema_period=9)
    enriched = logic.build_ema_trend_with_indicators(five_minute, config)

    test_minute = minute_history[minute_history["timestamp"].dt.date.between(args.start_date, args.end_date)].copy()
    if test_minute.empty:
        raise RuntimeError("Fetched SENSEX history contains no minute candles in the test date range.")

    results: dict[float, pd.DataFrame] = {}
    for cap in args.caps:
        trades = replay_sensex_touch_cap(
            enriched,
            test_minute,
            start=args.start_date,
            end=args.end_date,
            daily_loss_cap=cap,
            ema9_buffer_points=args.ema9_buffer_points,
            signal_logic=logic,
            strategy_config=config,
        )
        results[cap] = trades
        metrics = _trade_metrics(trades, cap, args.ema9_buffer_points)
        _, summary_path = _write_report(
            trades,
            start=args.start_date,
            end=args.end_date,
            metrics=metrics,
            minute_rows=len(test_minute),
            five_minute_bars=len(five_minute),
            cache=cache,
        )
        print(f"\n{summary_path.read_text(encoding='utf-8')}")

    if 250.0 in results:
        print(_compare_saved_baseline(results[250.0]))
    if 250.0 in results and 300.0 in results:
        delta = results[300.0]["index_points"].sum() - results[250.0]["index_points"].sum()
        print(f"Paired net delta (300 minus 250): {delta:+.2f} directional points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
