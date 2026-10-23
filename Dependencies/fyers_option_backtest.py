"""Backtest EMA index signals using Fyers historical NIFTY option candles."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import dotenv_values

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Dependencies.fyers_api import FyersAPI, FyersAPIError  # noqa: E402
from Dependencies.fyers_options import (  # noqa: E402
    FyersOptionContract,
    select_atm_option,
)

ENV_FILE = ROOT_DIR / "Dependencies" / ".env"
OUTPUT_DIR = ROOT_DIR / "Backtest Outputs"
IST = ZoneInfo("Asia/Kolkata")
UNDERLYING_SYMBOL = "NSE:NIFTY50-INDEX"
LOGGER = logging.getLogger("fyers_option_backtest")
OPTION_RESOLUTION_SECONDS = {"5S": 5, "1": 60}


@dataclass(slots=True)
class _OpenTrade:
    contract: FyersOptionContract
    direction: str
    action: str
    quantity: int
    entry_index_price: float
    entry_timestamp: pd.Timestamp
    entry_option_price: float
    option_bars: pd.DataFrame
    next_option_index: int
    option_resolution: str
    option_bar_seconds: int


def _completed_five_minute_bars(minute_bars: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(minute_bars.columns):
        raise ValueError("Fyers index history is missing required OHLCV columns.")
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
    bars = frame.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    counts = frame["close"].resample("5min", label="left", closed="left").count()
    bars = bars[(counts == 5) & bars["close"].notna()]
    return bars.reset_index(names="timestamp")


def _expiry_from_active_row(value: Any) -> date:
    return datetime.fromtimestamp(int(value), tz=ZoneInfo("UTC")).date()


def _archived_option_rows(symbols: list[str], expiry: date, lot_size: int) -> list[dict[str, Any]]:
    rows = []
    prefixes = (
        f"NIFTY{expiry:%y}{expiry.month}{expiry:%d}",
        f"NIFTY{expiry:%y}{expiry.strftime('%b')[0].upper()}{expiry:%d}",
        f"NIFTY{expiry:%y}{expiry.strftime('%b').upper()}",
    )
    for symbol in symbols:
        normalized = symbol.strip().upper()
        contract_name = normalized.rsplit(":", 1)[-1]
        right = contract_name[-2:] if contract_name.endswith(("CE", "PE")) else ""
        prefix = next((item for item in prefixes if contract_name.startswith(item)), None)
        if prefix is None or right not in {"CE", "PE"}:
            continue
        strike_text = contract_name[len(prefix) : -2]
        if not strike_text.isdigit():
            continue
        rows.append(
            {
                "symbol": symbol,
                "option_type": right,
                "strike_price": float(strike_text),
                "lot_size": lot_size,
                "expiry": expiry,
            }
        )
    return rows


def _second_expiry_on_or_after(expiries: list[date], trade_date: date) -> date:
    upcoming = sorted({expiry for expiry in expiries if expiry >= trade_date})
    if len(upcoming) < 2:
        raise ValueError(f"Fyers returned fewer than two expiries on or after {trade_date}.")
    return upcoming[1]


def _date_chunks(start: date, end: date, *, chunk_days: int = 90) -> list[tuple[date, date]]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")
    chunks = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks


def _fetch_index_history(api: FyersAPI, start: date, end: date) -> pd.DataFrame:
    frames = [
        api.history(UNDERLYING_SYMBOL, resolution="1", start=chunk_start, end=chunk_end)
        for chunk_start, chunk_end in _date_chunks(start, end)
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    history = pd.concat(frames, ignore_index=True)
    return history.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _fetch_expiry_dates(api: FyersAPI, start: date, end: date) -> list[date]:
    expiries = set()
    for chunk_start, chunk_end in _date_chunks(start, end):
        expiries.update(api.expired_option_expiries(UNDERLYING_SYMBOL, chunk_start, chunk_end))
    return sorted(expiries)


def _expired_data_query_end(end: date, today: date) -> date:
    return min(end + timedelta(days=21), today - timedelta(days=1))


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
    expiry = _second_expiry_on_or_after(expiries, trade_date)
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
    return select_atm_option(rows, action=action, spot=spot, expiry=expiry)


def _first_fill(bars: pd.DataFrame, timestamp: pd.Timestamp) -> tuple[int, pd.Series] | None:
    candidates = bars.index[bars["timestamp"] >= timestamp]
    if len(candidates) == 0:
        return None
    index = int(candidates[0])
    row = bars.iloc[index]
    price = float(row["open"])
    if not pd.notna(price) or price <= 0:
        return None
    return index, row


def _trade_record(
    trade: _OpenTrade,
    *,
    exit_timestamp: pd.Timestamp,
    exit_index_price: float,
    exit_option_price: float,
    exit_reason: str,
    exit_option_index: int,
) -> dict[str, Any]:
    window = trade.option_bars.iloc[trade.next_option_index : exit_option_index + 1]
    if window.empty:
        window = trade.option_bars.iloc[[max(0, exit_option_index)]]
    high = pd.to_numeric(window["high"], errors="coerce").dropna()
    low = pd.to_numeric(window["low"], errors="coerce").dropna()
    quantity = trade.quantity
    gross_pnl = (exit_option_price - trade.entry_option_price) * quantity
    return {
        "entry_time": trade.entry_timestamp.isoformat(),
        "exit_time": exit_timestamp.isoformat(),
        "signal": trade.action,
        "exit_reason": exit_reason,
        "symbol": trade.contract.symbol,
        "expiry": trade.contract.expiry.isoformat(),
        "option_type": trade.contract.option_type,
        "strike": trade.contract.strike,
        "lot_size": trade.contract.lot_size,
        "quantity": quantity,
        "entry_index": trade.entry_index_price,
        "exit_index": exit_index_price,
        "entry_option": trade.entry_option_price,
        "exit_option": exit_option_price,
        "gross_pnl": gross_pnl,
        "option_resolution": trade.option_resolution,
        "max_favorable_pnl": (float(high.max()) - trade.entry_option_price) * quantity if not high.empty else 0.0,
        "max_adverse_pnl": (float(low.min()) - trade.entry_option_price) * quantity if not low.empty else 0.0,
        "option_candles": len(window),
    }


def replay_index_signals(
    *,
    signal_logic: Any,
    strategy_config: Any,
    index_bars: pd.DataFrame,
    api: FyersAPI,
    active_rows: list[dict[str, Any]],
    expiries: list[date],
    archived_symbols: dict[date, list[str]],
    lots: int = 1,
    max_daily_loss: float = 5500.0,
    trade_start: date | None = None,
    trade_end: date | None = None,
    option_resolution: str = "5S",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if option_resolution not in OPTION_RESOLUTION_SECONDS:
        raise ValueError(f"Unsupported option resolution: {option_resolution}")
    option_bar_seconds = OPTION_RESOLUTION_SECONDS[option_resolution]
    enriched = signal_logic.build_ema_trend_with_indicators(index_bars, strategy_config)
    engine = signal_logic.EMATrendSignalEngine(strategy_config)
    contracts = [row for row in active_rows if row.get("underlying") == "NIFTY" and row.get("option_type") == "CE"]
    lot_size = int(contracts[0]["lot_size"]) if contracts else 65
    history_cache: dict[tuple[str, date], pd.DataFrame] = {}
    trades: list[dict[str, Any]] = []
    counters = {"signals": 0, "entries": 0, "exits": 0, "missing_option_data": 0, "daily_loss_halts": 0}
    position: _OpenTrade | None = None
    realized_by_day: dict[date, float] = {}
    halted_days: set[date] = set()

    def option_day(contract: FyersOptionContract, day: date) -> pd.DataFrame:
        key = (contract.symbol, day)
        if key not in history_cache:
            history_cache[key] = api.history(
                contract.symbol,
                resolution=option_resolution,
                start=day,
                end=day,
                expired=contract.expiry < date.today(),
            )
        result = history_cache[key].copy()
        if not result.empty:
            result = result.sort_values("timestamp").reset_index(drop=True)
        return result

    for index in range(len(enriched)):
        candle = enriched.iloc[index]
        candle_time = pd.Timestamp(candle["timestamp"])
        candle_time = (
            candle_time.tz_localize(IST) if candle_time.tzinfo is None else candle_time.tz_convert(IST)
        )
        day = candle_time.date()
        candle_close_time = candle_time + pd.Timedelta(minutes=5)
        session_time = candle_close_time.timetz().replace(tzinfo=None)
        if session_time > time(15, 15):
            continue

        position_context = None
        if position is not None:
            position_context = signal_logic.EMATrendPositionContext(
                direction=position.direction,
                entry_underlying=position.entry_index_price,
                stop_underlying=float(candle.get("ema9", 0.0)),
            )
        decision = engine.evaluate_candle(enriched.iloc[: index + 1], position=position_context)
        if (trade_start is not None and day < trade_start) or (trade_end is not None and day > trade_end):
            continue

        if position is not None:
            option_bars = position.option_bars
            while position.next_option_index < len(option_bars):
                option_index = position.next_option_index
                option_bar = option_bars.iloc[option_index]
                option_bar_end = pd.Timestamp(option_bar["timestamp"]) + pd.Timedelta(
                    seconds=position.option_bar_seconds
                )
                if option_bar_end > candle_close_time:
                    break
                position.next_option_index += 1
                option_mark = float(option_bar["close"])
                day_realized = realized_by_day.get(day, 0.0)
                if day_realized + (option_mark - position.entry_option_price) * position.quantity <= -max_daily_loss:
                    trade = _trade_record(
                        position,
                        exit_timestamp=option_bar_end,
                        exit_index_price=float(candle["close"]),
                        exit_option_price=option_mark,
                        exit_reason="DAILY_MAX_LOSS",
                        exit_option_index=option_index,
                    )
                    trades.append(trade)
                    realized_by_day[day] = day_realized + trade["gross_pnl"]
                    halted_days.add(day)
                    counters["daily_loss_halts"] += 1
                    counters["exits"] += 1
                    position = None
                    break

            if position is None:
                continue

            square_off = session_time >= time(15, 15)
            if decision.action == "EXIT" or square_off:
                reason = "SQUARE_OFF" if square_off else decision.exit_reason
                fill = _first_fill(position.option_bars, candle_close_time)
                if fill is None:
                    counters["missing_option_data"] += 1
                    position = None
                    continue
                option_index, option_bar = fill
                option_price = float(option_bar["open"])
                trade = _trade_record(
                    position,
                    exit_timestamp=pd.Timestamp(option_bar["timestamp"]),
                    exit_index_price=float(candle["close"]),
                    exit_option_price=option_price,
                    exit_reason=reason,
                    exit_option_index=option_index,
                )
                trades.append(trade)
                realized_by_day[day] = realized_by_day.get(day, 0.0) + trade["gross_pnl"]
                counters["exits"] += 1
                position = None
            continue

        if decision.signal_triggered:
            counters["signals"] += 1
        if day in halted_days or decision.action not in {"ENTER_LONG", "ENTER_SHORT"}:
            continue
        if not (time(9, 25) <= session_time < time(15, 15)):
            continue
        if realized_by_day.get(day, 0.0) <= -max_daily_loss:
            halted_days.add(day)
            continue

        action = decision.action
        try:
            contract = _resolve_contract(
                api,
                action=action,
                spot=float(candle["close"]),
                trade_date=day,
                active_rows=active_rows,
                expiries=expiries,
                archived_symbols=archived_symbols,
                fallback_lot_size=lot_size,
            )
            option_bars = option_day(contract, day)
        except (FyersAPIError, ValueError) as exc:
            LOGGER.warning("Skipping %s %s option data: %s", candle_close_time, action, exc)
            counters["missing_option_data"] += 1
            continue
        fill = _first_fill(option_bars, candle_close_time)
        if fill is None:
            counters["missing_option_data"] += 1
            continue
        option_index, option_bar = fill
        direction = "LONG" if action == "ENTER_LONG" else "SHORT"
        position = _OpenTrade(
            contract=contract,
            direction=direction,
            action=action,
            quantity=contract.lot_size * lots,
            entry_index_price=float(candle["close"]),
            entry_timestamp=pd.Timestamp(option_bar["timestamp"]),
            entry_option_price=float(option_bar["open"]),
            option_bars=option_bars,
            next_option_index=option_index + 1,
            option_resolution=option_resolution,
            option_bar_seconds=option_bar_seconds,
        )
        counters["entries"] += 1

    if position is not None:
        counters["missing_option_data"] += 1
    return trades, counters


def run_backtest(
    api: FyersAPI,
    signal_logic: Any,
    strategy_config: Any,
    start: date,
    end: date,
    *,
    lots: int = 1,
    max_daily_loss: float = 5500.0,
    option_resolution: str = "5S",
) -> tuple[pd.DataFrame, dict[str, int]]:
    index_minutes = _fetch_index_history(api, start - timedelta(days=7), end)
    index_bars = _completed_five_minute_bars(index_minutes)
    if index_bars.empty:
        raise RuntimeError("Fyers returned no complete NIFTY five-minute index candles for the requested period.")

    active_rows = api.option_master()
    expiry_rows = [
        _expiry_from_active_row(row["expiry"])
        for row in active_rows
        if row.get("underlying") == "NIFTY" and row.get("option_type") in {"CE", "PE"}
    ]
    expiry_query_end = _expired_data_query_end(end, date.today())
    archived_expiries = _fetch_expiry_dates(api, start, expiry_query_end)
    expiries = sorted(set(expiry_rows + archived_expiries))
    archived_symbols: dict[date, list[str]] = {}
    trades, counters = replay_index_signals(
        signal_logic=signal_logic,
        strategy_config=strategy_config,
        index_bars=index_bars,
        api=api,
        active_rows=active_rows,
        expiries=expiries,
        archived_symbols=archived_symbols,
        lots=lots,
        max_daily_loss=max_daily_loss,
        trade_start=start,
        trade_end=end,
        option_resolution=option_resolution,
    )
    if counters["missing_option_data"]:
        raise RuntimeError(
            "Backtest incomplete: Fyers option candles/contracts were missing for "
            f"{counters['missing_option_data']} signal/exit events; no report was written."
        )
    return pd.DataFrame(trades), counters


def _write_report(
    trades: pd.DataFrame,
    counters: dict[str, int],
    start: date,
    end: date,
    option_resolution: str,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"nifty_ema_fyers_options_{start:%Y%m%d}_{end:%Y%m%d}"
    trades_path = OUTPUT_DIR / f"{stem}_trades.csv"
    summary_path = OUTPUT_DIR / f"{stem}_summary.txt"
    trades.to_csv(trades_path, index=False)

    pnl = pd.to_numeric(trades.get("gross_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    gross = float(pnl.sum())
    profit_factor = float(winners.sum() / abs(losers.sum())) if not losers.empty else float("inf")
    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax() if not pnl.empty else pd.Series(dtype=float)
    lines = [
        f"Fyers NIFTY EMA option backtest: {start} through {end}",
        "Signals use NIFTY index 5-minute bars; option fills use the first option candle "
        "open after the completed signal bar.",
        f"Option candle resolution: {option_resolution}",
        "Expiry policy: second Fyers-listed expiry on or after each signal date.",
        "Expired-contract API returns symbols without historical lot size; quantity uses current Fyers NIFTY lot size.",
        "P&L is gross; brokerage, taxes, bid-ask spread, and slippage are not included.",
        f"Signals: {counters['signals']}",
        f"Entries: {counters['entries']}",
        f"Exits: {counters['exits']}",
        f"Trades completed: {len(trades)}",
        f"Missing option-data events: {counters['missing_option_data']}",
        f"Daily max-loss halts: {counters['daily_loss_halts']}",
        f"Win rate: {(len(winners) / len(pnl) * 100.0) if len(pnl) else 0.0:.2f}%",
        f"Gross option P&L: {gross:.2f}",
        f"Profit factor: {profit_factor:.4f}",
        f"Max trade-equity drawdown: {float(drawdown.min()) if not drawdown.empty else 0.0:.2f}",
        f"Trades CSV: {trades_path}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backtest EMA index signals against Fyers historical option candles."
    )
    parser.add_argument("--start-date", default="2026-10-01", help="Inclusive date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default="2026-10-09", help="Inclusive date in YYYY-MM-DD format.")
    parser.add_argument("--lots", type=int, default=1)
    parser.add_argument("--max-daily-loss", type=float, default=5500.0)
    parser.add_argument(
        "--option-resolution",
        choices=("auto", "5S", "1"),
        default="auto",
        help="auto uses 5S for recent history and 1-minute candles for older dates.",
    )
    args = parser.parse_args(argv)
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start > end or args.lots <= 0 or args.max_daily_loss <= 0:
        parser.error("date range must be valid and lots/loss limit must be positive")
    option_resolution = args.option_resolution
    if option_resolution == "auto":
        option_resolution = "5S" if start >= date.today() - timedelta(days=35) else "1"

    config = dotenv_values(ENV_FILE)
    app_id = str(config.get("FYERS_APP_ID", "")).strip()
    access_token = str(config.get("FYERS_ACCESS_TOKEN", "")).strip()
    if not app_id or not access_token:
        parser.error("Set FYERS_APP_ID and FYERS_ACCESS_TOKEN in Dependencies/.env")

    sys_path = str(ROOT_DIR / "Signal Generators")
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
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
        trades, counters = run_backtest(
            api,
            sys.modules["ema_trend_strategy_logic"],
            strategy_config,
            start,
            end,
            lots=args.lots,
            max_daily_loss=args.max_daily_loss,
            option_resolution=option_resolution,
        )
    except (FyersAPIError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    _write_report(trades, counters, start, end, option_resolution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
