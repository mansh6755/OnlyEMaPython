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
    _active_expiry,
    _history_chunks,
    _index_history_warmup_start,
    _load_signal_logic,
    _prepare_option_quotes,
    _quote_at,
)
from Dependencies.fyers_market_data import completed_five_minute_bars  # noqa: E402
from Dependencies.fyers_options import (  # noqa: E402
    FyersOptionContract,
    select_atm_option,
    select_nearest_expiry,
)
from Dependencies.fyers_telegram_auth import load_persisted_access_token  # noqa: E402

INDEX_SYMBOL = "BSE:SENSEX-INDEX"
DEFAULT_START = date(2020, 1, 1)
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
) -> tuple[pd.DataFrame, FyersHistoryCache, FyersAPI]:
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
    return minute_bars, cache, api


def _archived_sensex_option_rows(
    symbols: list[str],
    expiry: date,
    lot_size: int,
) -> list[dict[str, Any]]:
    month = expiry.strftime("%b").upper()
    prefixes = (
        f"SENSEX{expiry:%y}{expiry.month}{expiry:%d}",
        f"SENSEX{expiry:%y}{expiry.month:02d}{expiry:%d}",
        f"SENSEX{expiry:%y}{month[0]}{expiry:%d}",
        f"SENSEX{expiry:%y}{month}{expiry:%d}",
        f"SENSEX{expiry:%y}{month}",
    )
    rows = []
    for symbol in symbols:
        name = str(symbol).strip().upper().rsplit(":", 1)[-1]
        option_type = name[-2:] if name.endswith(("CE", "PE")) else ""
        prefix = next((candidate for candidate in prefixes if name.startswith(candidate)), None)
        if prefix is None or option_type not in {"CE", "PE"}:
            continue
        strike_text = name[len(prefix) : -2]
        try:
            strike = float(strike_text)
        except ValueError:
            continue
        if not math.isfinite(strike) or strike <= 0:
            continue
        rows.append(
            {
                "symbol": symbol,
                "option_type": option_type,
                "strike_price": strike,
                "lot_size": lot_size,
                "expiry": expiry,
                "symbol_prefix": symbol[: len(symbol) - len(strike_text) - len(option_type)],
            }
        )
    return rows


def _select_option_strike(
    options: list[dict[str, Any]],
    *,
    action: str,
    spot: float,
    expiry: date,
    moneyness_steps: int,
    strike_step_points: float | None = None,
) -> FyersOptionContract:
    """Select the requested listed strike steps ITM or OTM from nearest ATM."""
    direction = str(action).strip().upper()
    option_type = {"ENTER_LONG": "CE", "LONG": "CE", "ENTER_SHORT": "PE", "SHORT": "PE"}.get(direction)
    if option_type is None:
        raise ValueError(f"Unsupported EMA entry action: {action!r}")

    archived_prefix = next(
        (str(row.get("symbol_prefix", "")) for row in options if row.get("symbol_prefix")),
        "",
    )
    if archived_prefix and strike_step_points is not None:
        if not math.isfinite(strike_step_points) or strike_step_points <= 0:
            raise ValueError("strike_step_points must be positive and finite.")
        anchor_strike = float(next(row["strike_price"] for row in options if row.get("symbol_prefix")))
        nearest_grid_step = math.ceil((spot - anchor_strike) / strike_step_points - 0.5)
        atm_strike = anchor_strike + nearest_grid_step * strike_step_points
        strike_direction = -1 if option_type == "CE" else 1
        target_strike = atm_strike + strike_direction * moneyness_steps * strike_step_points
        if target_strike <= 0:
            raise ValueError("Selected SENSEX option strike must be positive.")
        lot_size = int(next(row["lot_size"] for row in options if row.get("symbol_prefix")))
        synthetic_contract = {
            "symbol": f"{archived_prefix}{target_strike:g}{option_type}",
            "option_type": option_type,
            "strike_price": target_strike,
            "lot_size": lot_size,
        }
        return select_atm_option(
            [synthetic_contract],
            action=action,
            spot=spot,
            expiry=expiry,
        )

    atm = select_atm_option(options, action=action, spot=spot, expiry=expiry)
    if moneyness_steps == 0:
        return atm

    valid_rows: list[tuple[dict[str, Any], float]] = []
    for row in options:
        if str(row.get("option_type", "")).strip().upper() != option_type:
            continue
        try:
            strike = float(row.get("strike_price"))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(strike) and strike > 0:
            valid_rows.append((row, strike))
    strikes = sorted({strike for _, strike in valid_rows})
    atm_index = strikes.index(atm.strike)
    strike_direction = -1 if option_type == "CE" else 1
    target_index = atm_index + strike_direction * moneyness_steps
    if target_index < 0 or target_index >= len(strikes):
        label = "ITM" if moneyness_steps > 0 else "OTM"
        raise ValueError(
            f"No {abs(moneyness_steps)}-step {label} {option_type} contract is listed "
            f"for expiry {expiry}."
        )
    target_strike = strikes[target_index]
    return select_atm_option(
        [
            row
            for row, strike in valid_rows
            if strike == target_strike
        ],
        action=action,
        spot=spot,
        expiry=expiry,
    )


def _attach_option_prices(
    trades: pd.DataFrame,
    *,
    api: FyersAPI,
    cache: FyersHistoryCache,
    start: date,
    end: date,
    resolution: str,
    option_daily_loss_cap: float | None = None,
    minute_bars: pd.DataFrame | None = None,
    option_strike_steps: int = 0,
) -> pd.DataFrame:
    """Add ATM option marks and optionally enforce a daily option-P&L cap."""
    if option_daily_loss_cap is not None and (
        not math.isfinite(option_daily_loss_cap) or option_daily_loss_cap <= 0
    ):
        raise ValueError("option_daily_loss_cap must be positive and finite.")
    if option_daily_loss_cap is not None and minute_bars is None:
        raise ValueError("minute_bars are required to mark SENSEX points at an option-cap exit.")

    result = trades.copy()
    columns = [
        "option_symbol",
        "option_expiry",
        "option_type",
        "option_strike",
        "option_strike_steps",
        "option_lot_size",
        "option_resolution",
        "entry_option_price",
        "entry_option_candle_time",
        "exit_option_price",
        "exit_option_candle_time",
        "estimated_option_pnl",
        "option_price_status",
        "index_exit_mark_basis",
    ]
    for column in columns:
        result[column] = pd.NA
    if result.empty:
        return result

    minute_closes: pd.DataFrame | None = None
    if minute_bars is not None:
        minute_closes = minute_bars[["timestamp", "close"]].copy()
        minute_closes["timestamp"] = pd.to_datetime(minute_closes["timestamp"], errors="coerce")
        if minute_closes["timestamp"].dt.tz is None:
            minute_closes["timestamp"] = minute_closes["timestamp"].dt.tz_localize(IST)
        else:
            minute_closes["timestamp"] = minute_closes["timestamp"].dt.tz_convert(IST)
        minute_closes["close"] = pd.to_numeric(minute_closes["close"], errors="coerce")
        minute_closes = minute_closes.dropna(subset=["timestamp", "close"])
        minute_closes["completed_at"] = minute_closes["timestamp"] + pd.Timedelta(minutes=1)
        minute_closes = minute_closes.sort_values("completed_at").reset_index(drop=True)

    active_rows = [
        row
        for row in api.option_master("BSE")
        if row.get("underlying") == "SENSEX" and row.get("option_type") in {"CE", "PE"}
    ]
    if not active_rows:
        raise RuntimeError("Fyers BSE derivatives master has no SENSEX option contracts.")
    fallback_lot_size = int(active_rows[0]["lot_size"])
    current_strikes = sorted(
        {
            float(row["strike_price"])
            for row in active_rows
            if row.get("strike_price") is not None and math.isfinite(float(row["strike_price"]))
        }
    )
    strike_gaps = [float(gap) for gap in np.diff(current_strikes) if gap > 0]
    archived_strike_step = min(strike_gaps) if strike_gaps else None
    result.attrs["archived_strike_step_points"] = archived_strike_step
    query_end = min(end + timedelta(days=7), date.today() - timedelta(days=1))
    expiries: set[date] = set()
    for chunk_start, chunk_end in _history_chunks(start, query_end) if start <= query_end else []:
        expiries.update(api.expired_option_expiries(INDEX_SYMBOL, chunk_start, chunk_end))
    for row in active_rows:
        row_expiry = _active_expiry(row)
        if row_expiry is None:
            continue
        if row_expiry >= date.today():
            expiries.add(row_expiry)
    if not expiries:
        raise RuntimeError("Fyers returned no SENSEX option expiries for the selected replay dates.")

    archived_symbols: dict[date, list[str]] = {}
    option_quote_cache: dict[tuple[str, date], Any] = {}
    daily_option_realized: dict[date, float] = {}
    halted_days: set[date] = set()
    skipped_indices: list[Any] = []

    for index, trade in result.iterrows():
        trade_day = pd.Timestamp(trade["entry_time"]).date()
        if trade_day in halted_days:
            skipped_indices.append(index)
            continue

        expiry = select_nearest_expiry(expiries, today=trade_day)
        if expiry < date.today():
            if expiry not in archived_symbols:
                archived_symbols[expiry] = api.expired_option_symbols(INDEX_SYMBOL, expiry)
            contract_rows = _archived_sensex_option_rows(
                archived_symbols[expiry],
                expiry,
                fallback_lot_size,
            )
        else:
            contract_rows = [
                {**row, "expiry": expiry}
                for row in active_rows
                if _active_expiry(row) == expiry
                and row.get("option_type") in {"CE", "PE"}
            ]
        contract = _select_option_strike(
            contract_rows,
            action=str(trade["signal"]),
            spot=float(trade["entry_index"]),
            expiry=expiry,
            moneyness_steps=option_strike_steps,
            strike_step_points=archived_strike_step,
        )
        key = (contract.symbol, trade_day)
        if key not in option_quote_cache:
            history = cache.history(
                api,
                contract.symbol,
                resolution=resolution,
                start=trade_day,
                end=trade_day,
                expired=expiry < date.today(),
            )
            option_quote_cache[key] = _prepare_option_quotes(history, resolution)
        quotes = option_quote_cache[key]
        entry_quote = _quote_at(quotes, pd.Timestamp(trade["entry_time"]), resolution)
        exit_quote = _quote_at(quotes, pd.Timestamp(trade["exit_time"]), resolution)
        result.at[index, "option_symbol"] = contract.symbol
        result.at[index, "option_expiry"] = contract.expiry.isoformat()
        result.at[index, "option_type"] = contract.option_type
        result.at[index, "option_strike"] = contract.strike
        result.at[index, "option_strike_steps"] = option_strike_steps
        result.at[index, "option_lot_size"] = contract.lot_size
        result.at[index, "option_resolution"] = resolution
        if entry_quote is None:
            if option_daily_loss_cap is not None:
                raise RuntimeError(
                    f"Cannot enforce the option daily cap: no completed 5-second entry mark "
                    f"for {contract.symbol} at {trade['entry_time']}."
                )
            missing = []
            if exit_quote is None:
                missing.extend(("ENTRY", "EXIT"))
            else:
                missing.append("ENTRY")
            result.at[index, "option_price_status"] = (
                f"MISSING_{'_AND_'.join(missing)}_CANDLE" if missing else "MISSING_ENTRY_CANDLE"
            )
            continue

        result.at[index, "entry_option_price"] = entry_quote[0]
        result.at[index, "entry_option_candle_time"] = entry_quote[1].isoformat()
        exit_time = pd.Timestamp(trade["exit_time"])
        exit_price = exit_quote[0] if exit_quote is not None else None
        exit_candle_time = exit_quote[1] if exit_quote is not None else None
        exit_reason = str(trade.get("exit_reason", ""))

        if option_daily_loss_cap is not None:
            realized = daily_option_realized.get(trade_day, 0.0)
            entry_timestamp_ns = pd.Timestamp(trade["entry_time"]).value
            first_mark = int(
                np.searchsorted(quotes.completed_at_ns, entry_timestamp_ns, side="right")
            )
            last_mark = int(np.searchsorted(quotes.completed_at_ns, exit_time.value, side="right"))
            for mark_index in range(first_mark, last_mark):
                mark_price = float(quotes.close[mark_index])
                if not math.isfinite(mark_price) or mark_price <= 0:
                    continue
                open_pnl = (mark_price - entry_quote[0]) * contract.lot_size
                if realized + open_pnl <= -option_daily_loss_cap:
                    exit_price = mark_price
                    exit_candle_time = pd.Timestamp(quotes.completed_at_ns[mark_index], tz=IST)
                    exit_time = exit_candle_time
                    exit_reason = "OPTION_DAILY_MAX_LOSS"
                    halted_days.add(trade_day)
                    if minute_closes is None:
                        raise RuntimeError("Minute index data is required for an option-cap exit.")
                    completed_index = minute_closes[minute_closes["completed_at"] <= exit_time]
                    if completed_index.empty:
                        index_exit = float(trade["entry_index"])
                        index_mark_basis = "ENTRY_INDEX_NO_COMPLETED_MINUTE"
                    else:
                        index_exit = float(completed_index.iloc[-1]["close"])
                        index_mark_basis = "LAST_COMPLETED_1M_CLOSE"
                    direction = 1.0 if str(trade["signal"]) == "ENTER_LONG" else -1.0
                    result.at[index, "exit_time"] = exit_time.isoformat()
                    result.at[index, "exit_reason"] = exit_reason
                    result.at[index, "exit_index"] = index_exit
                    result.at[index, "index_points"] = direction * (
                        index_exit - float(trade["entry_index"])
                    )
                    result.at[index, "index_exit_mark_basis"] = index_mark_basis
                    break

        if exit_price is None or exit_candle_time is None:
            if option_daily_loss_cap is not None:
                raise RuntimeError(
                    f"Cannot complete the option-cap backtest: no completed 5-second exit mark "
                    f"for {contract.symbol} at or before {trade['exit_time']}."
                )
            result.at[index, "option_price_status"] = "MISSING_EXIT_CANDLE"
            continue

        result.at[index, "exit_option_price"] = exit_price
        result.at[index, "exit_option_candle_time"] = exit_candle_time.isoformat()
        option_pnl = (exit_price - entry_quote[0]) * contract.lot_size
        result.at[index, "estimated_option_pnl"] = option_pnl
        result.at[index, "option_price_status"] = "OK"
        if option_daily_loss_cap is not None:
            daily_option_realized[trade_day] = daily_option_realized.get(trade_day, 0.0) + option_pnl
        if pd.isna(result.at[index, "index_exit_mark_basis"]):
            result.at[index, "index_exit_mark_basis"] = "INDEX_REPLAY_FILL"

    if skipped_indices:
        result = result.drop(index=skipped_indices)
    return result


def replay_sensex_touch_cap(
    enriched_bars: pd.DataFrame,
    minute_bars: pd.DataFrame,
    *,
    start: date,
    end: date,
    daily_loss_cap: float | None,
    ema9_buffer_points: float = 3.0,
    signal_logic: Any,
    strategy_config: Any,
) -> pd.DataFrame:
    """Replay EMA9 touch exits with five-minute-close trail and daily-cap checks."""
    if daily_loss_cap is not None and (not math.isfinite(daily_loss_cap) or daily_loss_cap <= 0):
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
        if daily_loss_cap is not None and daily_realized_points <= -daily_loss_cap:
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
                    minute_close_points = direction * (
                        float(minute.close) - position["entry_price"]
                    )
                    if daily_loss_cap is not None and daily_realized_points + minute_close_points <= -daily_loss_cap:
                        close_position(minute_close_time, float(minute.close), "DAILY_MAX_LOSS")
                        break

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

                if daily_loss_cap is not None and daily_realized_points + close_points <= -daily_loss_cap:
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
            or (daily_loss_cap is not None and daily_realized_points <= -daily_loss_cap)
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


def _trade_metrics(
    trades: pd.DataFrame,
    cap: float,
    ema9_buffer_points: float = 0.0,
    *,
    cap_mode: str = "index",
) -> dict[str, Any]:
    points = pd.to_numeric(trades["index_points"], errors="raise")
    equity = points.cumsum()
    # Fixed drawdown calculation to yield a positive magnitude
    drawdown = equity.cummax() - equity
    cap_reason = "OPTION_DAILY_MAX_LOSS" if cap_mode == "option" else "DAILY_MAX_LOSS"
    cap_exit_days = trades.loc[trades["exit_reason"] == cap_reason, "exit_time"].str[:10].nunique()
    metrics = {
        "trades": len(trades),
        "wins": int((points > 0).sum()),
        "losses": int((points < 0).sum()),
        "win_rate": float((points > 0).mean() * 100.0) if len(points) else 0.0,
        "net_points": float(points.sum()),
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "cap_hit_days": int(cap_exit_days),
        "exit_counts": trades["exit_reason"].value_counts().to_dict(),
        "cap": cap,
        "cap_mode": cap_mode,
        "ema9_buffer_points": ema9_buffer_points,
    }
    if cap_mode == "option":
        option_points = pd.to_numeric(trades["estimated_option_pnl"], errors="raise")
        option_equity = option_points.cumsum()
        option_drawdown = option_equity.cummax() - option_equity
        daily_option_points = trades.assign(
            trade_day=pd.to_datetime(trades["entry_time"]).dt.date,
            option_pnl=option_points,
        ).groupby("trade_day")["option_pnl"].sum()
        gross_profit = float(option_points[option_points > 0].sum())
        gross_loss = float(option_points[option_points < 0].sum())
        metrics.update(
            {
                "option_wins": int((option_points > 0).sum()),
                "option_losses": int((option_points < 0).sum()),
                "option_win_rate": float((option_points > 0).mean() * 100.0) if len(option_points) else 0.0,
                "estimated_option_pnl": float(option_points.sum()),
                "max_option_trade_drawdown": float(option_drawdown.max()) if len(option_drawdown) else 0.0,
                "max_daily_option_pnl": float(daily_option_points.max()) if len(daily_option_points) else 0.0,
                "worst_daily_option_pnl": float(daily_option_points.min()) if len(daily_option_points) else 0.0,
                "average_option_trade_pnl": float(option_points.mean()) if len(option_points) else 0.0,
                "option_profit_factor": gross_profit / abs(gross_loss) if gross_loss else math.inf,
            }
        )
    return metrics


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
    cap_label = (
        f"optioncap{metrics['cap']:g}"
        if metrics.get("cap_mode") == "option"
        else f"cap{metrics['cap']:g}"
    )
    stem = f"fyers_ema_sensex_{start:%Y%m%d}_{end:%Y%m%d}_touch_{cap_label}_5m_close"
    if metrics["ema9_buffer_points"] > 0:
        stem += f"_ema9buffer{metrics['ema9_buffer_points']:g}"
    if metrics.get("option_prices_enabled"):
        stem += f"_optionprices{metrics['option_resolution']}"
    if metrics.get("option_strike_suffix"):
        stem += f"_strike_{metrics['option_strike_suffix']}"
    trades_path = output_dir / f"{stem}_trades.csv"
    summary_path = output_dir / f"{stem}_summary.txt"
    trades.to_csv(trades_path, index=False)
    monthly = trades.assign(exit_month=pd.to_datetime(trades["exit_time"]).dt.strftime("%Y-%m"))
    monthly_points = monthly.groupby("exit_month")["index_points"].sum()
    lines = [
        f"SENSEX EMA9-touch replay, {start} through {end}",
        (
            "Exploratory reconstructed replay with a rupee-denominated option cap."
            if metrics.get("cap_mode") == "option"
            else (
                "Exploratory reconstructed replay; compare the 250-point result against the saved baseline "
                "before treating the cap comparison as verified."
            )
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
            (
                f"Daily option cap: aggregate realized plus open one-lot gross option P&L checked at each "
                f"completed five-second candle; exit at that candle close at or below -₹{metrics['cap']:g} "
                "and halt entries for the day. Index points for option-cap exits use the latest completed "
                "one-minute SENSEX close available at the option exit timestamp."
            )
            if metrics.get("cap_mode") == "option"
            else (
                f"Daily cap: realized plus minute-close open points checked at each completed one-minute "
                f"close; exit at that close at or below -{metrics['cap']:g} points and halt entries for the day."
            )
        ),
        (
            "Entry at completed five-minute signal close from 09:25 IST; square-off at 15:15 IST. "
            "No 1R breakeven, fees, spread, slippage, latency, or market impact."
        ),
        (
            "Option prices: disabled; this is an index-only replay."
            if not metrics.get("option_prices_enabled")
            else (
                f"{metrics.get('option_strike_label', 'ATM')} option prices use completed "
                f"{metrics['option_resolution']} candles available at or before each index entry/exit. "
                "Option P&L is a one-lot gross estimate based "
                "on candle closes; fees, spread, slippage, and historical lot-size changes are not modeled. "
                + (
                    f"Archived strikes are reconstructed using the current master spacing "
                    f"({metrics['archived_strike_step_points']:g} points). "
                    if metrics.get("archived_strike_step_points") is not None
                    else ""
                )
                + f"Estimated gross option P&L={metrics['estimated_option_pnl']:+.2f}; "
                f"price marks: {metrics['option_price_status_counts']}."
            )
        ),
        (
            f"Data: minute rows={minute_rows}; complete five-minute bars={five_minute_bars}; "
            f"cache hits={cache.hits}; API fetches={cache.misses}."
        ),
        (
            (
                f"Trades={metrics['trades']}; SENSEX-mark wins={metrics['wins']}; "
                f"losses={metrics['losses']}; win rate={metrics['win_rate']:.2f}%; "
                f"option wins={metrics['option_wins']}; losses={metrics['option_losses']}; "
                f"option win rate={metrics['option_win_rate']:.2f}%; "
                f"trade-equity max drawdown=₹{metrics['max_option_trade_drawdown']:.2f}; "
                f"worst daily option P&L=₹{metrics['worst_daily_option_pnl']:.2f}; "
                f"profit factor={metrics['option_profit_factor']:.3f}."
            )
            if metrics.get("cap_mode") == "option"
            else (
                f"Trades={metrics['trades']}; wins={metrics['wins']}; losses={metrics['losses']}; "
                f"win rate={metrics['win_rate']:.2f}%."
            )
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
    cap_group = parser.add_mutually_exclusive_group()
    cap_group.add_argument(
        "--caps",
        type=float,
        nargs="+",
        help="One or more daily loss caps in index points (default: 250 300).",
    )
    cap_group.add_argument(
        "--option-daily-loss-cap",
        type=float,
        help=(
            "Replace the index-point cap with an aggregate gross option-P&L cap in rupees, checked at "
            "each completed 5-second option candle; for example, --option-daily-loss-cap 3100."
        ),
    )
    parser.add_argument(
        "--ema9-buffer-points",
        type=float,
        default=3.0,
        help="Move LONG EMA9 stops down and SHORT EMA9 stops up by this many points (default: 3).",
    )
    parser.add_argument(
        "--option-prices",
        action="store_true",
        help="Capture nearest-expiry option prices using Fyers Prime 5-second candles.",
    )
    parser.add_argument(
        "--option-strike-steps",
        type=int,
        nargs="+",
        default=[0],
        metavar="STEPS",
        help=(
            "Select listed strikes relative to nearest ATM: 0=ATM, positive=ITM steps, "
            "negative=OTM steps (CE/PE direction handled automatically). May specify several "
            "values for a side-by-side replay."
        ),
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--refresh-cache", action="store_true", help="Fetch again and replace cached requests.")
    cache_group.add_argument("--no-cache", action="store_true", help="Fetch without reading or writing local cache.")
    args = parser.parse_args(argv)
    if args.start_date > args.end_date:
        parser.error("--start-date must not be after --end-date.")
    caps = (
        args.caps
        if args.caps is not None
        else ([] if args.option_daily_loss_cap is not None else list(DEFAULT_CAPS))
    )
    if args.option_daily_loss_cap is not None and (
        not math.isfinite(args.option_daily_loss_cap) or args.option_daily_loss_cap <= 0
    ):
        parser.error("--option-daily-loss-cap must be a positive finite rupee amount.")
    if args.option_daily_loss_cap is None and (
        not caps or any(not math.isfinite(cap) or cap <= 0 for cap in caps)
    ):
        parser.error("--caps values must be positive finite point amounts.")
    if len(set(caps)) != len(caps):
        parser.error("--caps must not contain duplicates.")
    if not math.isfinite(args.ema9_buffer_points) or args.ema9_buffer_points < 0:
        parser.error("--ema9-buffer-points must be finite and non-negative.")
    if len(set(args.option_strike_steps)) != len(args.option_strike_steps):
        parser.error("--option-strike-steps must not contain duplicates.")
    if any(args.option_strike_steps) and not (args.option_prices or args.option_daily_loss_cap is not None):
        parser.error("--option-strike-steps requires --option-prices or --option-daily-loss-cap.")

    minute_history, cache, api = fetch_minute_history(
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

    option_resolution = "5S"
    option_cap_mode = args.option_daily_loss_cap is not None
    caps_to_run = [args.option_daily_loss_cap] if option_cap_mode else caps
    strike_steps_to_run = args.option_strike_steps if args.option_prices or option_cap_mode else [0]
    strike_comparison_rows: list[dict[str, Any]] = []
    for cap in caps_to_run:
        trades = replay_sensex_touch_cap(
            enriched,
            test_minute,
            start=args.start_date,
            end=args.end_date,
            daily_loss_cap=None if option_cap_mode else cap,
            ema9_buffer_points=args.ema9_buffer_points,
            signal_logic=logic,
            strategy_config=config,
        )
        for strike_steps in strike_steps_to_run:
            option_trades = trades
            if args.option_prices or option_cap_mode:
                option_trades = _attach_option_prices(
                    trades,
                    api=api,
                    cache=cache,
                    start=args.start_date,
                    end=args.end_date,
                    resolution=option_resolution,
                    option_daily_loss_cap=args.option_daily_loss_cap,
                    minute_bars=test_minute,
                    option_strike_steps=strike_steps,
                )
            metrics = _trade_metrics(
                option_trades,
                cap,
                args.ema9_buffer_points,
                cap_mode="option" if option_cap_mode else "index",
            )
            metrics["option_prices_enabled"] = args.option_prices or option_cap_mode
            metrics["option_strike_steps"] = strike_steps
            metrics["option_strike_label"] = (
                "ATM"
                if strike_steps == 0
                else f"{abs(strike_steps)}-step {'ITM' if strike_steps > 0 else 'OTM'}"
            )
            if args.option_prices or option_cap_mode:
                metrics["option_resolution"] = option_resolution
                metrics["archived_strike_step_points"] = option_trades.attrs.get(
                    "archived_strike_step_points"
                )
                metrics["option_price_status_counts"] = (
                    option_trades["option_price_status"].value_counts(dropna=False).to_dict()
                )
                metrics["estimated_option_pnl"] = float(
                    pd.to_numeric(option_trades["estimated_option_pnl"], errors="coerce").sum()
                )
            if option_cap_mode:
                metrics["option_daily_loss_cap"] = cap
            if strike_steps != 0 or len(strike_steps_to_run) > 1:
                metrics["option_strike_suffix"] = (
                    "ATM"
                    if strike_steps == 0
                    else f"{abs(strike_steps)}step{'ITM' if strike_steps > 0 else 'OTM'}"
                )
            _, summary_path = _write_report(
                option_trades,
                start=args.start_date,
                end=args.end_date,
                metrics=metrics,
                minute_rows=len(test_minute),
                five_minute_bars=len(five_minute),
                cache=cache,
            )
            print(f"\n{summary_path.read_text(encoding='utf-8')}")
            if len(strike_steps_to_run) > 1:
                option_statuses = option_trades["option_price_status"].value_counts()
                strike_comparison_rows.append(
                    {
                        "selection": metrics["option_strike_label"],
                        "moneyness_steps": strike_steps,
                        "trades": metrics["trades"],
                        "gross_option_pnl": metrics["estimated_option_pnl"],
                        "max_trade_equity_drawdown": metrics["max_option_trade_drawdown"],
                        "worst_daily_option_pnl": metrics["worst_daily_option_pnl"],
                        "best_daily_option_pnl": metrics["max_daily_option_pnl"],
                        "win_rate_pct": metrics["option_win_rate"],
                        "profit_factor": metrics["option_profit_factor"],
                        "cap_hit_days": metrics["cap_hit_days"],
                        "priced_trades": int(option_statuses.get("OK", 0)),
                        "missing_price_trades": int(len(option_trades) - option_statuses.get("OK", 0)),
                    }
                )

    if strike_comparison_rows:
        comparison = pd.DataFrame(strike_comparison_rows)
        output_dir = ROOT_DIR / "Backtest Outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        cap_label = f"optioncap{args.option_daily_loss_cap:g}" if option_cap_mode else "optionprices"
        comparison_path = output_dir / (
            f"fyers_ema_sensex_{args.start_date:%Y%m%d}_{args.end_date:%Y%m%d}_"
            f"strike_comparison_{cap_label}_{option_resolution}.csv"
        )
        comparison.to_csv(comparison_path, index=False)
        print("\nStrike-selection comparison (gross, one lot):")
        print(comparison.to_string(index=False))
        print(f"Comparison CSV: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
