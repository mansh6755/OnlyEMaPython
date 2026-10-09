"""Paper-first EMA runner backed by Fyers candles and NIFTY option contracts."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as day_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from Dependencies.broker_contract import OrderStatus
from Dependencies.fyers_api import FyersAPI, FyersAPIError
from Dependencies.fyers_execution import FyersExecutionClient
from Dependencies.fyers_options import FyersOptionContract

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / "Dependencies" / ".env"
IST = ZoneInfo("Asia/Kolkata")
LOGGER = logging.getLogger("fyers_ema_runner")


@dataclass(slots=True)
class ActivePosition:
    direction: str
    contract: FyersOptionContract
    quantity: int
    entry_underlying: float
    stop_underlying: float
    entry_option_price: float


def completed_five_minute_bars(minute_bars: pd.DataFrame, now: datetime) -> pd.DataFrame:
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
    bars = frame.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    counts = frame["close"].resample("5min", label="left", closed="left").count()
    bars = bars[(counts == 5) & bars["close"].notna()]
    return bars.reset_index(names="timestamp")


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def daily_loss_limit_reached(
    realized_pnl: float,
    entry_price: float,
    mark_price: float,
    quantity: int,
    max_daily_loss: float,
) -> bool:
    unrealized_pnl = (mark_price - entry_price) * quantity
    return realized_pnl + unrealized_pnl <= -max_daily_loss


def _entry_bar_is_fresh(timestamp: pd.Timestamp, now: datetime) -> bool:
    bar_close = timestamp + pd.Timedelta(minutes=5)
    bar_close_time = bar_close.to_pydatetime()
    return timestamp.date() == now.date() and timedelta(0) <= now - bar_close_time <= timedelta(minutes=2)


def _close_position(
    api: FyersAPI,
    execution: FyersExecutionClient,
    position: ActivePosition,
    *,
    live_enabled: bool,
    product_type: str,
    reason: str,
) -> float:
    if live_enabled:
        result = execution.place_market_order(
            position.contract.symbol,
            "SELL",
            position.quantity,
            product_type=product_type,
            order_tag="EMA-EXIT",
        )
        if result.status is not OrderStatus.FILLED:
            raise RuntimeError("Fyers exit was not fully confirmed; stop and reconcile the account.")
        exit_price = result.average_fill_price
        if exit_price <= 0:
            exit_price = api.quote_ltp(position.contract.symbol)
    else:
        exit_price = api.quote_ltp(position.contract.symbol)
    trade_pnl = (exit_price - position.entry_option_price) * position.quantity
    LOGGER.info(
        "EMA exit confirmed | symbol=%s | reason=%s | pnl=%.2f",
        position.contract.symbol,
        reason,
        trade_pnl,
    )
    return trade_pnl


def _live_preflight(client: FyersExecutionClient) -> None:
    if not client.ensure_logged_in():
        raise RuntimeError("Fyers login failed; live execution is disabled for this run.")
    if not client.preload_scrip_master():
        raise RuntimeError("Fyers contract master could not be loaded; live execution is disabled.")
    orders = client.list_open_orders()
    positions = client.list_open_positions()
    if orders.is_indeterminate or positions.is_indeterminate:
        raise RuntimeError("Fyers account state is indeterminate; live execution is blocked.")
    if orders.items or positions.items:
        raise RuntimeError("Fyers account is not flat; resolve open orders/positions before live startup.")


def run_ema_runner(signal_logic: Any, strategy_config: Any) -> None:
    load_dotenv(ENV_FILE, override=False)
    app_id = os.getenv("FYERS_APP_ID", "").strip()
    access_token = os.getenv("FYERS_ACCESS_TOKEN", "").strip()
    underlying_symbol = os.getenv("FYERS_UNDERLYING_SYMBOL", "NSE:NIFTY50-INDEX").strip()
    live_enabled = os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() == "true"
    lots = _positive_int("EMA_LOTS", 1)
    poll_seconds = _positive_int("EMA_POLL_SECONDS", 30)
    max_daily_loss = float(os.getenv("EMA_MAX_LOSS", "5500"))
    if not math.isfinite(max_daily_loss) or max_daily_loss <= 0:
        raise ValueError("EMA_MAX_LOSS must be finite and positive.")
    if not app_id or not access_token:
        raise RuntimeError("Set FYERS_APP_ID and FYERS_ACCESS_TOKEN in Dependencies/.env first.")
    if os.getenv("EMA_OPTION_EXPIRY_POLICY", "NEXT_WEEK").strip().upper() != "NEXT_WEEK":
        raise RuntimeError("This runner currently supports EMA_OPTION_EXPIRY_POLICY=NEXT_WEEK only.")
    if os.getenv("EMA_OPTION_STRIKE_POLICY", "ATM").strip().upper() != "ATM":
        raise RuntimeError("This runner currently supports EMA_OPTION_STRIKE_POLICY=ATM only.")

    api = FyersAPI(app_id, access_token)
    execution = FyersExecutionClient(api)
    try:
        profile = api.profile()
    except FyersAPIError as exc:
        raise RuntimeError("Fyers authentication failed; refresh the access token.") from exc
    if profile.get("s") != "ok":
        raise RuntimeError("Fyers authentication failed; refresh the access token.")
    if live_enabled:
        _live_preflight(execution)

    signal_engine = signal_logic.EMATrendSignalEngine(strategy_config)
    position: ActivePosition | None = None
    realized_pnl = 0.0
    active_date: date | None = None
    last_processed: pd.Timestamp | None = None
    LOGGER.info("EMA runner connected to Fyers; mode=%s; lots=%s", "LIVE" if live_enabled else "PAPER", lots)
    LOGGER.info("Option policy: next week's listed expiry, ATM, buy CE/PE by EMA direction.")

    while True:
        now = datetime.now(IST)
        if active_date != now.date():
            active_date = now.date()
            realized_pnl = 0.0
            last_processed = None
        if now.timetz().replace(tzinfo=None) >= day_time(15, 15):
            if position is not None:
                try:
                    realized_pnl += _close_position(
                        api,
                        execution,
                        position,
                        live_enabled=live_enabled,
                        product_type=os.getenv("EMA_PRODUCT_TYPE", "INTRADAY"),
                        reason="SQUARE_OFF",
                    )
                except (FyersAPIError, RuntimeError) as exc:
                    LOGGER.critical("Scheduled EMA exit unresolved: %s", exc)
                    raise
            LOGGER.info("EMA session ended at the configured square-off time.")
            return
        try:
            minute_bars = api.history(
                underlying_symbol,
                resolution="1",
                start=now.date() - timedelta(days=7),
                end=now.date(),
            )
        except FyersAPIError as exc:
            LOGGER.error("Fyers market-data refresh failed; retrying: %s", exc)
            time.sleep(poll_seconds)
            continue

        bars = completed_five_minute_bars(minute_bars, now)
        if bars.empty:
            time.sleep(poll_seconds)
            continue
        strategy_frame = signal_logic.build_ema_trend_with_indicators(bars, strategy_config)
        if strategy_frame.empty:
            time.sleep(poll_seconds)
            continue

        current = strategy_frame.iloc[-1]
        current_timestamp = pd.Timestamp(current["timestamp"])
        entry_window = (
            day_time(9, 25) <= now.timetz().replace(tzinfo=None) < day_time(15, 15)
            and _entry_bar_is_fresh(current_timestamp, now)
        )
        square_off_due = now.timetz().replace(tzinfo=None) >= day_time(15, 15)
        position_context = None
        if position is not None:
            position_context = signal_logic.EMATrendPositionContext(
                direction=position.direction,
                entry_underlying=position.entry_underlying,
                stop_underlying=position.stop_underlying,
            )
        decision = (
            signal_engine.evaluate_candle(strategy_frame, position=position_context)
            if current_timestamp != last_processed
            else signal_logic.EMATrendDecision()
        )
        mark_price: float | None = None
        risk_exit_due = False
        if position is not None:
            try:
                mark_price = api.quote_ltp(position.contract.symbol)
                risk_exit_due = daily_loss_limit_reached(
                    realized_pnl,
                    position.entry_option_price,
                    mark_price,
                    position.quantity,
                    max_daily_loss,
                )
            except FyersAPIError as exc:
                LOGGER.warning("Unable to refresh the open option mark: %s", exc)

        if position is not None and (square_off_due or risk_exit_due or decision.action == "EXIT"):
            try:
                exit_price = mark_price or api.quote_ltp(position.contract.symbol)
                if live_enabled:
                    result = execution.place_market_order(
                        position.contract.symbol,
                        "SELL",
                        position.quantity,
                        product_type=os.getenv("EMA_PRODUCT_TYPE", "INTRADAY"),
                        order_tag="EMA-EXIT",
                    )
                    if result.status is not OrderStatus.FILLED:
                        raise RuntimeError(
                            "Fyers exit was not fully confirmed; stop and reconcile the account."
                        )
                    if result.average_fill_price > 0:
                        exit_price = result.average_fill_price
                trade_pnl = (exit_price - position.entry_option_price) * position.quantity
                realized_pnl += trade_pnl
                LOGGER.info(
                    "EMA exit confirmed | symbol=%s | reason=%s | pnl=%.2f | daily=%.2f",
                    position.contract.symbol,
                    "SQUARE_OFF"
                    if square_off_due
                    else "DAILY_MAX_LOSS"
                    if risk_exit_due
                    else decision.exit_reason,
                    trade_pnl,
                    realized_pnl,
                )
                position = None
            except (FyersAPIError, RuntimeError) as exc:
                LOGGER.critical("EMA exit unresolved; halt and reconcile manually: %s", exc)
                raise
        elif position is None and decision.action in {"ENTER_LONG", "ENTER_SHORT"}:
            if realized_pnl <= -max_daily_loss:
                LOGGER.warning("Daily loss limit reached; new EMA entries are disabled.")
            elif entry_window:
                contract = api.resolve_next_week_option(
                    decision.action,
                    spot=float(current["close"]),
                    today=now.date(),
                )
                quantity = contract.lot_size * lots
                entry_price = api.quote_ltp(contract.symbol)
                if live_enabled:
                    result = execution.place_market_order(
                        contract.symbol,
                        "BUY",
                        quantity,
                        product_type=os.getenv("EMA_PRODUCT_TYPE", "INTRADAY"),
                        order_tag="EMA-ENTRY",
                    )
                    if result.status is OrderStatus.REJECTED:
                        LOGGER.warning("Fyers rejected EMA entry; no position opened.")
                    elif result.status is not OrderStatus.FILLED:
                        LOGGER.critical("EMA entry outcome is ambiguous; halt and reconcile manually.")
                        raise RuntimeError("Fyers entry was not fully confirmed.")
                    elif result.average_fill_price > 0:
                        entry_price = result.average_fill_price
                    else:
                        LOGGER.warning("Fyers confirmed a fill without an average price; using the last quote.")
                        entry_price = api.quote_ltp(contract.symbol)
                if not live_enabled or result.status is OrderStatus.FILLED:
                    direction = "LONG" if decision.action == "ENTER_LONG" else "SHORT"
                    position = ActivePosition(
                        direction,
                        contract,
                        quantity,
                        float(current["close"]),
                        float(decision.stop_underlying),
                        entry_price,
                    )
                    LOGGER.info(
                        "%s EMA entry | action=%s | symbol=%s | expiry=%s | strike=%s | qty=%s | ltp=%.2f",
                        "LIVE" if live_enabled else "PAPER",
                        decision.action,
                        contract.symbol,
                        contract.expiry,
                        contract.strike,
                        quantity,
                        entry_price,
                    )

        if current_timestamp != last_processed:
            last_processed = current_timestamp
        time.sleep(poll_seconds)
