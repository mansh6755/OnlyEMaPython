"""Backtest NIFTY EMA9-touch exits against historical Fyers option candles."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import talib
from dotenv import dotenv_values

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Dependencies.fyers_api import FyersAPI, FyersAPIError  # noqa: E402
from Dependencies.fyers_option_backtest import (  # noqa: E402
    OUTPUT_DIR,
    UNDERLYING_SYMBOL,
    _archived_option_rows,
    _completed_five_minute_bars,
    _date_chunks,
    _expired_data_query_end,
    _expiry_from_active_row,
    _fetch_expiry_dates,
)
from Dependencies.fyers_options import (  # noqa: E402
    FyersOptionContract,
    select_atm_option,
    select_nearest_expiry,
)
from Dependencies.fyers_history_cache import FyersHistoryCache, HISTORY_COLUMNS  # noqa: E402

ENV_FILE = ROOT_DIR / "Dependencies" / ".env"
IST = ZoneInfo("Asia/Kolkata")
ENTRY_START = time(9, 25)
SQUARE_OFF = time(15, 15)
OPTION_RESOLUTION = "5S"
OPTION_BAR_SECONDS = 5
LOGGER = logging.getLogger("fyers_nifty_ema9_touch_backtest")


def replay_ema9_touch_index(
    *,
    signal_logic: Any,
    strategy_config: Any,
    five_minute_bars: pd.DataFrame,
    minute_bars: pd.DataFrame,
    start: date,
    end: date,
    ema9_buffer_points: float = 3.0,
) -> list[dict[str, Any]]:
    """Replay original NIFTY entries with the SENSEX EMA9 touch/trailing exits."""
    if not np.isfinite(ema9_buffer_points) or ema9_buffer_points < 0:
        raise ValueError("ema9_buffer_points must be finite and non-negative.")
    if five_minute_bars.empty:
        raise ValueError("No complete NIFTY five-minute bars are available.")
    enriched = signal_logic.build_ema_trend_with_indicators(five_minute_bars, strategy_config)
    enriched = enriched.copy()
    enriched["ema9"] = talib.EMA(
        pd.to_numeric(enriched["close"], errors="coerce").to_numpy(dtype=float),
        timeperiod=9,
    )

    bar_times = pd.to_datetime(enriched["timestamp"], errors="coerce")
    bar_times = bar_times.dt.tz_localize(IST) if bar_times.dt.tz is None else bar_times.dt.tz_convert(IST)
    if bar_times.isna().any():
        raise ValueError("NIFTY five-minute bars contain invalid timestamps.")

    minutes = minute_bars.copy()
    if "timestamp" not in minutes:
        raise ValueError("NIFTY minute data is missing its timestamp column.")
    minutes["timestamp"] = pd.to_datetime(minutes["timestamp"], errors="coerce")
    if minutes["timestamp"].dt.tz is None:
        minutes["timestamp"] = minutes["timestamp"].dt.tz_localize(IST)
    else:
        minutes["timestamp"] = minutes["timestamp"].dt.tz_convert(IST)
    minutes = minutes.dropna(subset=["timestamp"])
    minutes = minutes[minutes["timestamp"].dt.date.between(start, end)].copy()
    minutes["bar_start"] = minutes["timestamp"].dt.floor("5min")
    minutes_by_bar = {key: group for key, group in minutes.groupby("bar_start", sort=True)}

    trades: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None
    engine = signal_logic.EMATrendSignalEngine(strategy_config)

    def close_position(at: pd.Timestamp, price: float, reason: str) -> None:
        nonlocal position
        if position is None:
            raise RuntimeError("Cannot close an absent NIFTY replay position.")
        direction = 1.0 if position["signal"] == "ENTER_LONG" else -1.0
        trades.append(
            {
                "signal_bar_start": position["signal_bar_start"].isoformat(),
                "entry_time": position["entry_time"].isoformat(),
                "exit_time": at.isoformat(),
                "signal": position["signal"],
                "exit_reason": reason,
                "entry_index": position["entry_price"],
                "exit_index": price,
                "index_points": direction * (price - position["entry_price"]),
            }
        )
        position = None

    for bar_index, bar in enumerate(enriched.itertuples(index=False)):
        bar_start = bar_times.iloc[bar_index]
        trading_day = bar_start.date()
        if not start <= trading_day <= end:
            continue
        bar_close_time = bar_start + pd.Timedelta(minutes=5)
        position_open_at_bar_start = position is not None
        minute_group = minutes_by_bar.get(bar_start)

        if position is not None and bar_index > position["entry_bar_index"]:
            prior_ema9 = float(enriched.iloc[bar_index - 1]["ema9"])
            direction = 1.0 if position["signal"] == "ENTER_LONG" else -1.0
            prior_stop = prior_ema9 - direction * ema9_buffer_points
            if minute_group is not None and np.isfinite(prior_stop):
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

            if position is not None:
                direction = 1.0 if position["signal"] == "ENTER_LONG" else -1.0
                trailing_stop = position["trailing_stop"]
                trail_hit = (
                    float(bar.close) <= trailing_stop
                    if direction > 0
                    else float(bar.close) >= trailing_stop
                )
                if trail_hit:
                    close_position(bar_close_time, float(bar.close), "TRAILING_STOP")
                elif bar_close_time.time() >= SQUARE_OFF:
                    close_position(bar_close_time, float(bar.close), "SQUARE_OFF")
                else:
                    current_ema9 = float(bar.ema9)
                    if np.isfinite(current_ema9):
                        buffered_ema9 = current_ema9 - direction * ema9_buffer_points
                        position["trailing_stop"] = (
                            max(trailing_stop, buffered_ema9)
                            if direction > 0
                            else min(trailing_stop, buffered_ema9)
                        )

        if position is not None or position_open_at_bar_start:
            continue
        if bar_close_time.time() >= SQUARE_OFF or bar_close_time.time() < ENTRY_START:
            continue
        decision = engine.evaluate_candle(enriched.iloc[: bar_index + 1])
        if decision.action not in {"ENTER_LONG", "ENTER_SHORT"}:
            continue
        entry_ema9 = float(bar.ema9)
        if not np.isfinite(entry_ema9):
            continue
        direction = 1.0 if decision.action == "ENTER_LONG" else -1.0
        position = {
            "signal_bar_start": bar_start,
            "entry_time": bar_close_time,
            "signal": decision.action,
            "entry_price": float(bar.close),
            "entry_bar_index": bar_index,
            "trailing_stop": entry_ema9 - direction * ema9_buffer_points,
        }

    if position is not None:
        raise RuntimeError("NIFTY EMA9 replay ended with an open position.")
    return trades


def _fetch_index_history_with_cache(
    api: FyersAPI,
    start: date,
    end: date,
    history_cache: FyersHistoryCache,
) -> pd.DataFrame:
    frames = [
        history_cache.history(
            api,
            UNDERLYING_SYMBOL,
            resolution="1",
            start=chunk_start,
            end=chunk_end,
        )
        for chunk_start, chunk_end in _date_chunks(start, end)
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def _select_itm_contract(
    api: FyersAPI,
    *,
    action: str,
    spot: float,
    trade_day: date,
    active_rows: list[dict[str, Any]],
    expiries: list[date],
    archived_symbols: dict[date, list[str]],
    fallback_lot_size: int,
    strike_steps: int,
) -> FyersOptionContract:
    expiry = select_nearest_expiry(expiries, today=trade_day)
    if expiry < date.today():
        if expiry not in archived_symbols:
            archived_symbols[expiry] = api.expired_option_symbols(UNDERLYING_SYMBOL, expiry)
        rows = _archived_option_rows(archived_symbols[expiry], expiry, fallback_lot_size)
    else:
        rows = []
        for row in active_rows:
            if row.get("underlying") != "NIFTY" or row.get("option_type") not in {"CE", "PE"}:
                continue
            try:
                row_expiry = _expiry_from_active_row(row.get("expiry"))
            except (TypeError, ValueError, OverflowError, OSError):
                continue
            if row_expiry == expiry:
                rows.append({**row, "expiry": expiry})

    atm = select_atm_option(rows, action=action, spot=spot, expiry=expiry)
    option_type = atm.option_type
    strikes = sorted(
        {
            float(row["strike_price"])
            for row in rows
            if str(row.get("option_type", "")).upper() == option_type
        }
    )
    atm_index = strikes.index(atm.strike)
    target_index = atm_index + (-strike_steps if option_type == "CE" else strike_steps)
    if target_index < 0 or target_index >= len(strikes):
        raise ValueError(
            f"No {strike_steps}-step ITM {option_type} contract is listed for {expiry}."
        )
    target_strike = strikes[target_index]
    return select_atm_option(
        [
            row
            for row in rows
            if str(row.get("option_type", "")).upper() == option_type
            and float(row["strike_price"]) == target_strike
        ],
        action=action,
        spot=spot,
        expiry=expiry,
    )


def _completed_option_quotes(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=["completed_at", "open", "high", "low", "close"])
    quotes = bars.copy()
    quotes["timestamp"] = pd.to_datetime(quotes["timestamp"], errors="coerce")
    if quotes["timestamp"].dt.tz is None:
        quotes["timestamp"] = quotes["timestamp"].dt.tz_localize(IST)
    else:
        quotes["timestamp"] = quotes["timestamp"].dt.tz_convert(IST)
    quotes["completed_at"] = quotes["timestamp"] + pd.Timedelta(seconds=OPTION_BAR_SECONDS)
    for column in ("open", "high", "low", "close"):
        quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
    return (
        quotes.dropna(subset=["completed_at", "open", "high", "low", "close"])
        .sort_values("completed_at")
        .drop_duplicates("completed_at", keep="last")
        .reset_index(drop=True)
    )


def _quote_index_at_or_before(quotes: pd.DataFrame, timestamp: pd.Timestamp) -> int:
    if quotes.empty:
        return -1
    return int(quotes["completed_at"].searchsorted(pd.Timestamp(timestamp), side="right")) - 1


def attach_option_prices(
    trades: list[dict[str, Any]],
    *,
    api: FyersAPI,
    active_rows: list[dict[str, Any]],
    expiries: list[date],
    lots: int,
    daily_loss_cap: float,
    start: date,
    end: date,
    history_cache: FyersHistoryCache,
    strike_steps: int = 2,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mark trades at completed 5-second closes and enforce the rupee loss cap."""
    if lots <= 0 or strike_steps <= 0:
        raise ValueError("lots and strike_steps must be positive.")
    if not np.isfinite(daily_loss_cap) or daily_loss_cap <= 0:
        raise ValueError("daily_loss_cap must be finite and positive.")

    fallback_lot_size = next(
        (
            int(row["lot_size"])
            for row in active_rows
            if row.get("underlying") == "NIFTY"
            and row.get("option_type") == "CE"
            and row.get("lot_size")
        ),
        65,
    )
    archived_symbols: dict[date, list[str]] = {}
    option_cache: dict[tuple[str, date], pd.DataFrame] = {}
    daily_realized: dict[date, float] = {}
    halted_days: set[date] = set()
    results: list[dict[str, Any]] = []
    counters = {"missing_option_data": 0, "daily_loss_halts": 0}

    for trade in trades:
        entry_time = pd.Timestamp(trade["entry_time"])
        exit_time = pd.Timestamp(trade["exit_time"])
        trade_day = entry_time.date()
        if not start <= trade_day <= end or trade_day in halted_days:
            continue
        try:
            contract = _select_itm_contract(
                api,
                action=trade["signal"],
                spot=float(trade["entry_index"]),
                trade_day=trade_day,
                active_rows=active_rows,
                expiries=expiries,
                archived_symbols=archived_symbols,
                fallback_lot_size=fallback_lot_size,
                strike_steps=strike_steps,
            )
            cache_key = (contract.symbol, trade_day)
            if cache_key not in option_cache:
                history = history_cache.history(
                    api,
                    contract.symbol,
                    resolution=OPTION_RESOLUTION,
                    start=trade_day,
                    end=trade_day,
                    expired=contract.expiry < date.today(),
                )
                option_cache[cache_key] = _completed_option_quotes(history)
        except (FyersAPIError, ValueError) as exc:
            LOGGER.warning("Missing %s option data for %s: %s", trade["signal"], entry_time, exc)
            counters["missing_option_data"] += 1
            continue

        quotes = option_cache[cache_key]
        completed_at = quotes["completed_at"]
        entry_index = _quote_index_at_or_before(quotes, entry_time)
        if entry_index < 0:
            counters["missing_option_data"] += 1
            continue
        entry_price = float(quotes.iloc[entry_index]["close"])
        quantity = contract.lot_size * lots
        realized = daily_realized.get(trade_day, 0.0)
        first_mark = int(completed_at.searchsorted(entry_time, side="right"))
        last_mark = int(completed_at.searchsorted(exit_time, side="right"))
        cap_index: int | None = None
        for quote_index in range(first_mark, last_mark):
            mark_price = float(quotes.iloc[quote_index]["close"])
            if realized + (mark_price - entry_price) * quantity <= -daily_loss_cap:
                cap_index = quote_index
                break

        if cap_index is None:
            exit_index = _quote_index_at_or_before(quotes, exit_time)
            if exit_index < 0:
                counters["missing_option_data"] += 1
                continue
            exit_price = float(quotes.iloc[exit_index]["close"])
            final_exit_time = exit_time
            final_exit_reason = trade["exit_reason"]
        else:
            exit_index = cap_index
            exit_price = float(quotes.iloc[exit_index]["close"])
            final_exit_time = pd.Timestamp(quotes.iloc[exit_index]["completed_at"])
            final_exit_reason = "OPTION_DAILY_MAX_LOSS"
            halted_days.add(trade_day)
            counters["daily_loss_halts"] += 1

        option_pnl = (exit_price - entry_price) * quantity
        daily_realized[trade_day] = realized + option_pnl
        record = {
            **trade,
            "exit_time": final_exit_time.isoformat(),
            "exit_reason": final_exit_reason,
            "option_symbol": contract.symbol,
            "expiry": contract.expiry.isoformat(),
            "option_type": contract.option_type,
            "strike": contract.strike,
            "strike_steps_itm": strike_steps,
            "lot_size": contract.lot_size,
            "quantity": quantity,
            "entry_option": entry_price,
            "entry_option_mark_time": quotes.iloc[entry_index]["completed_at"].isoformat(),
            "exit_option": exit_price,
            "exit_option_mark_time": quotes.iloc[exit_index]["completed_at"].isoformat(),
            "gross_pnl": option_pnl,
            "option_resolution": OPTION_RESOLUTION,
        }
        results.append(record)
    if counters["missing_option_data"]:
        raise RuntimeError(
            "Backtest incomplete: option data/contracts were missing for "
            f"{counters['missing_option_data']} trade events; no report should be used."
        )
    return pd.DataFrame(results), counters


def run_backtest(
    api: FyersAPI,
    signal_logic: Any,
    strategy_config: Any,
    start: date,
    end: date,
    *,
    lots: int = 1,
    daily_loss_cap: float = 5500.0,
    strike_steps: int = 2,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> tuple[pd.DataFrame, dict[str, int], int]:
    history_cache = FyersHistoryCache(
        OUTPUT_DIR / ".fyers_cache",
        enabled=use_cache,
        refresh=refresh_cache,
    )
    index_minutes = _fetch_index_history_with_cache(
        api,
        start - timedelta(days=7),
        end,
        history_cache,
    )
    five_minute_bars = _completed_five_minute_bars(index_minutes)
    index_trades = replay_ema9_touch_index(
        signal_logic=signal_logic,
        strategy_config=strategy_config,
        five_minute_bars=five_minute_bars,
        minute_bars=index_minutes,
        start=start,
        end=end,
    )
    active_rows = api.option_master()
    expiry_rows = [
        _expiry_from_active_row(row["expiry"])
        for row in active_rows
        if row.get("underlying") == "NIFTY"
        and row.get("option_type") in {"CE", "PE"}
        and row.get("expiry") is not None
    ]
    archived_expiries = _fetch_expiry_dates(
        api,
        start,
        _expired_data_query_end(end, date.today()),
    )
    expiries = sorted(set(expiry_rows + archived_expiries))
    trades, counters = attach_option_prices(
        index_trades,
        api=api,
        active_rows=active_rows,
        expiries=expiries,
        lots=lots,
        daily_loss_cap=daily_loss_cap,
        start=start,
        end=end,
        history_cache=history_cache,
        strike_steps=strike_steps,
    )
    counters["history_cache_hits"] = history_cache.hits
    counters["history_cache_misses"] = history_cache.misses
    return trades, counters, len(index_trades)


def write_report(
    trades: pd.DataFrame,
    counters: dict[str, int],
    index_trade_count: int,
    start: date,
    end: date,
    *,
    lots: int,
    daily_loss_cap: float,
    strike_steps: int,
) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = (
        f"nifty_ema9_touch_options_{start:%Y%m%d}_{end:%Y%m%d}_"
        f"buffer3_cap{daily_loss_cap:g}_{strike_steps}stepITM_{lots}lot"
    )
    trades_path = OUTPUT_DIR / f"{stem}_trades.csv"
    summary_path = OUTPUT_DIR / f"{stem}_summary.txt"
    trades.to_csv(trades_path, index=False)
    pnl = pd.to_numeric(trades.get("gross_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax().clip(lower=0)
    monthly = trades.copy()
    if not monthly.empty:
        monthly["month"] = pd.to_datetime(monthly["entry_time"]).dt.strftime("%Y-%m")
        monthly_pnl = monthly.groupby("month")["gross_pnl"].sum()
    else:
        monthly_pnl = pd.Series(dtype=float)
    lines = [
        f"NIFTY EMA9-touch option backtest: {start} through {end}",
        "Entries use the original NIFTY EMA4/11/18 signal engine on completed five-minute index bars.",
        "EMA9 touch exits use one-minute OHLC against the prior completed five-minute EMA9 with a 3-point buffer.",
        "The buffered EMA9 stop is ratcheted at each surviving five-minute close; square-off is 15:15 IST.",
        f"Option selection: {strike_steps}-step ITM from nearest listed ATM, nearest upcoming expiry.",
        f"Option resolution: {OPTION_RESOLUTION}; daily cap: INR {daily_loss_cap:.2f} gross realized plus open P&L.",
        "Daily cap is checked on completed five-second option candles; historical contracts use current lot size.",
        "P&L excludes brokerage, taxes, bid-ask spread, and slippage.",
        f"Index trades before option-data/cap processing: {index_trade_count}",
        f"Completed option trades: {len(trades)}",
        f"Missing option-data events: {counters['missing_option_data']}",
        f"Daily max-loss halts: {counters['daily_loss_halts']}",
        f"History cache hits/misses: {counters['history_cache_hits']}/{counters['history_cache_misses']}",
        f"Win rate: {(len(winners) / len(pnl) * 100.0) if len(pnl) else 0.0:.2f}%",
        f"Gross option P&L: {float(pnl.sum()):.2f}",
        f"Profit factor: {float(winners.sum() / abs(losers.sum())) if not losers.empty else float('inf'):.4f}",
        f"Max closed-trade equity drawdown: {float(drawdown.min()) if not drawdown.empty else 0.0:.2f}",
        "Monthly gross P&L:",
        *(f"{month}: {value:+.2f}" for month, value in monthly_pnl.items()),
        f"Trades CSV: {trades_path}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return trades_path, summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-10-23")
    parser.add_argument("--lots", type=int, default=1)
    parser.add_argument("--daily-loss-cap", type=float, default=5500.0)
    parser.add_argument("--strike-steps", type=int, default=2)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--no-cache", action="store_true", help="Disable persistent history caching.")
    cache_group.add_argument("--refresh-cache", action="store_true", help="Refresh cached history requests.")
    args = parser.parse_args(argv)
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start > end or args.lots <= 0 or args.daily_loss_cap <= 0 or args.strike_steps <= 0:
        parser.error("date range must be valid and lots, cap, and strike steps must be positive")

    config = dotenv_values(ENV_FILE)
    app_id = str(config.get("FYERS_APP_ID", "")).strip()
    access_token = str(config.get("FYERS_ACCESS_TOKEN", "")).strip()
    if not app_id or not access_token:
        parser.error("Set FYERS_APP_ID and FYERS_ACCESS_TOKEN in Dependencies/.env")

    signal_path = str(ROOT_DIR / "Signal Generators")
    if signal_path not in sys.path:
        sys.path.insert(0, signal_path)
    from ema_trend_strategy_logic import EMATrendConfig

    strategy_config = EMATrendConfig(
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
    )
    api = FyersAPI(app_id, access_token)
    try:
        trades, counters, index_trade_count = run_backtest(
            api,
            sys.modules["ema_trend_strategy_logic"],
            strategy_config,
            start,
            end,
            lots=args.lots,
            daily_loss_cap=args.daily_loss_cap,
            strike_steps=args.strike_steps,
            use_cache=not args.no_cache,
            refresh_cache=args.refresh_cache,
        )
    except (FyersAPIError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    write_report(
        trades,
        counters,
        index_trade_count,
        start,
        end,
        lots=args.lots,
        daily_loss_cap=args.daily_loss_cap,
        strike_steps=args.strike_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
