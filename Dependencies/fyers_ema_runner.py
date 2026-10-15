"""Paper-first EMA runner backed by Fyers candles and NIFTY option contracts."""

from __future__ import annotations

import logging
import math
import os
import time
import traceback
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
from Dependencies.fyers_dashboard_state import FyersDashboardState
from Dependencies.fyers_execution import FyersExecutionClient
from Dependencies.fyers_market_data import completed_five_minute_bars
from Dependencies.fyers_options import FyersOptionContract
from Dependencies.fyers_telegram_auth import (
    FyersTelegramAuthService,
    load_persisted_access_token,
)
from Dependencies.secret_redaction import environment_secrets, redact_text
from Dependencies.telegram_notifier import TelegramNotifier

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
    placed_at: datetime


@dataclass(frozen=True, slots=True)
class ClosedPosition:
    exit_option_price: float
    pnl: float
    placed_at: datetime


def _format_pnl(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}₹{abs(value):.2f}"


def _display_instrument(contract: FyersOptionContract) -> str:
    return f"NIFTY-{contract.expiry:%b%Y}-{contract.strike:g}-{contract.option_type}"


def _dashboard_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _dashboard_candles(strategy_frame: pd.DataFrame, *, limit: int = 240) -> list[dict[str, Any]]:
    candles: list[dict[str, Any]] = []
    for _, row in strategy_frame.tail(limit).iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        timestamp = timestamp.tz_localize(IST) if timestamp.tzinfo is None else timestamp.tz_convert(IST)
        candle = {
            "time": int(timestamp.timestamp()),
            "open": _dashboard_number(row.get("open")),
            "high": _dashboard_number(row.get("high")),
            "low": _dashboard_number(row.get("low")),
            "close": _dashboard_number(row.get("close")),
            "ema_fast": _dashboard_number(row.get("ema4")),
            "ema_mid": _dashboard_number(row.get("ema11")),
            "ema_slow": _dashboard_number(row.get("ema18")),
        }
        if all(candle[key] is not None for key in ("open", "high", "low", "close")):
            candles.append(candle)
    return candles


def _dashboard_position(position: ActivePosition | None, mark_price: float | None) -> dict[str, Any] | None:
    if position is None:
        return None
    mark = mark_price if mark_price is not None else position.entry_option_price
    return {
        "direction": position.direction,
        "symbol": position.contract.symbol,
        "display_symbol": _display_instrument(position.contract),
        "quantity": position.quantity,
        "entry_price": position.entry_option_price,
        "mark_price": mark,
        "open_pnl": (mark - position.entry_option_price) * position.quantity,
        "entry_spot": position.entry_underlying,
        "stop_spot": position.stop_underlying,
        "opened_at": position.placed_at.isoformat(),
    }


def _dashboard_closed_trade(
    position: ActivePosition,
    closed: ClosedPosition,
    reason: str,
    mode: str,
) -> dict[str, Any]:
    return {
        "time": position.placed_at.isoformat(),
        "exit_time": closed.placed_at.isoformat(),
        "direction": position.direction,
        "symbol": position.contract.symbol,
        "display_symbol": _display_instrument(position.contract),
        "quantity": position.quantity,
        "entry_price": position.entry_option_price,
        "exit_price": closed.exit_option_price,
        "pnl": closed.pnl,
        "reason": reason,
        "mode": mode,
    }


def _format_entry_alert(position: ActivePosition, lots: int, mode: str) -> str:
    contract = position.contract
    return (
        f"🚀 ENTRY · EMA\n"
        f"Direction: {position.direction}\n"
        f"Instrument: BUY {_display_instrument(contract)}\n"
        f"  Entry: ₹{position.entry_option_price:.2f}\n"
        f"Qty: {position.quantity} ({lots} lot(s) x {contract.lot_size})\n"
        f"[{mode}] {position.placed_at.astimezone(IST):%Y-%m-%d %H:%M:%S} IST"
    )


def _format_exit_alert(
    position: ActivePosition,
    closed: ClosedPosition,
    reason: str,
    mode: str,
) -> str:
    display_reason = "TIME_CUTOFF" if reason == "SQUARE_OFF" else reason
    return (
        f"🏁 EXIT · EMA ({display_reason})\n"
        f"Direction: {position.direction}\n"
        f"Instrument: SELL {_display_instrument(position.contract)}\n"
        f"  Entry: ₹{position.entry_option_price:.2f} → Exit: ₹{closed.exit_option_price:.2f}\n"
        f"Qty: {position.quantity} (lot size {position.contract.lot_size})\n"
        f"⚪ P&L: {_format_pnl(closed.pnl)}\n"
        f"[{mode}] {closed.placed_at.astimezone(IST):%Y-%m-%d %H:%M:%S} IST"
    )


def _format_eod_alert(realized_pnl: float, trade_count: int, mode: str, at: datetime) -> str:
    return (
        f"📊 End-of-Day P&L [{mode}]\n"
        f"{at.astimezone(IST):%Y-%m-%d %H:%M:%S} IST\n"
        f"EMA [{mode}]: {_format_pnl(realized_pnl)} ({trade_count} trade(s))\n"
        "──────────\n"
        f"Total: {_format_pnl(realized_pnl)} ({trade_count} trade(s))"
    )


def _format_error_alert(context: str, error: Exception, mode: str) -> str:
    details = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    details = redact_text(details, environment_secrets(os.environ)).strip()
    if len(details) > 2800:
        details = "[Earlier traceback details truncated]\n" + details[-2800:]
    return (
        f"🚨 ERROR · EMA\n"
        f"Mode: {mode}\n"
        f"Context: {context}\n"
        f"Time: {datetime.now(IST):%Y-%m-%d %H:%M:%S} IST\n\n"
        f"{details}"
    )


def _notify_runner_error(
    notifier: TelegramNotifier,
    context: str,
    error: Exception,
    mode: str,
) -> None:
    notifier.notify_error(
        f"{context}:{type(error).__name__}",
        _format_error_alert(context, error, mode),
    )


def _format_strategy_started_alert(mode: str, underlying_symbol: str, lots: int, at: datetime) -> str:
    return (
        f"✅ Strategy started successfully · EMA\n"
        f"Mode: {mode}\n"
        f"Underlying: {underlying_symbol}\n"
        f"Lots: {lots}\n"
        f"Time: {at.astimezone(IST):%Y-%m-%d %H:%M:%S} IST"
    )


def _format_day_started_alert(
    mode: str,
    trading_day: date,
    minute_bar_count: int,
    completed_bar_count: int,
    at: datetime,
) -> str:
    return (
        f"📅 Day started successfully · EMA\n"
        f"Mode: {mode}\n"
        f"Trading day: {trading_day.isoformat()}\n"
        f"Fyers API: fetched {minute_bar_count} one-minute bar(s)\n"
        f"Strategy data: {completed_bar_count} completed five-minute bar(s)\n"
        f"Time: {at.astimezone(IST):%Y-%m-%d %H:%M:%S} IST"
    )


def _format_health_alert(
    mode: str,
    position: ActivePosition | None,
    realized_pnl: float,
    market_data_status: str,
    last_market_data_at: datetime | None,
    at: datetime,
) -> str:
    position_status = (
        f"{position.direction} {_display_instrument(position.contract)} x {position.quantity}"
        if position is not None
        else "FLAT"
    )
    last_data = (
        f"{last_market_data_at.astimezone(IST):%Y-%m-%d %H:%M:%S} IST"
        if last_market_data_at is not None
        else "not received yet"
    )
    return (
        f"💓 HEALTH · EMA\n"
        f"Status: RUNNING\n"
        f"Mode: {mode}\n"
        f"Time: {at.astimezone(IST):%Y-%m-%d %H:%M:%S} IST\n"
        f"Market data: {market_data_status} (last success: {last_data})\n"
        f"Position: {position_status}\n"
        f"Realized P&L today: {_format_pnl(realized_pnl)}"
    )


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
    mark_price: float | None = None,
) -> ClosedPosition:
    if live_enabled:
        placed_at = datetime.now(IST)
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
            exit_price = mark_price or api.quote_ltp(position.contract.symbol)
    else:
        placed_at = datetime.now(IST)
        exit_price = mark_price or api.quote_ltp(position.contract.symbol)
    trade_pnl = (exit_price - position.entry_option_price) * position.quantity
    LOGGER.info(
        "EMA exit confirmed | symbol=%s | reason=%s | pnl=%.2f",
        position.contract.symbol,
        reason,
        trade_pnl,
    )
    return ClosedPosition(exit_price, trade_pnl, placed_at)


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
    notifier = TelegramNotifier.from_environment()
    mode = "LIVE" if os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() == "true" else "PAPER"
    auth_service: FyersTelegramAuthService | None = None
    dashboard_state = FyersDashboardState()
    dashboard_enabled = os.getenv("DASHBOARD_ENABLED", "true").strip().lower() == "true"
    try:
        app_id = os.getenv("FYERS_APP_ID", "").strip()
        token_file_value = os.getenv("FYERS_TOKEN_FILE", "").strip()
        token_file = (
            Path(token_file_value).expanduser()
            if token_file_value
            else ROOT_DIR / "Dependencies" / ".fyers_access_token"
        )
        if not token_file.is_absolute():
            token_file = ROOT_DIR / token_file
        access_token = load_persisted_access_token(
            token_file,
            os.getenv("FYERS_ACCESS_TOKEN", ""),
        )
        api = FyersAPI(app_id, access_token or "telegram-auth-pending")
        auth_service = FyersTelegramAuthService(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            authorized_chat_id=os.getenv("FYERS_AUTH_TELEGRAM_CHAT_ID", ""),
            client_id=app_id,
            secret_key=os.getenv("FYERS_SECRET_KEY", ""),
            redirect_uri=os.getenv("FYERS_REDIRECT_URI", ""),
            token_file=token_file,
            market_data_client=api,
            port=int(os.getenv("PORT", "8787")),
            dashboard_state=dashboard_state,
            dashboard_enabled=dashboard_enabled,
        )
        auth_service.start()
        auth_service.authenticate_startup(has_access_token=bool(access_token))
        dashboard_state.record_event(
            {"at": datetime.now(IST).isoformat(), "level": "SUCCESS", "message": "Fyers session authenticated"}
        )
        _run_ema_runner(signal_logic, strategy_config, notifier, api, dashboard_state)
    except Exception as exc:
        _notify_runner_error(notifier, "Unrecoverable runner error", exc, mode)
        notifier.close(timeout=10.0)
        raise
    finally:
        if auth_service is not None:
            auth_service.stop()


def _run_ema_runner(
    signal_logic: Any,
    strategy_config: Any,
    notifier: TelegramNotifier,
    api: FyersAPI,
    dashboard_state: FyersDashboardState,
) -> None:
    underlying_symbol = os.getenv("FYERS_UNDERLYING_SYMBOL", "NSE:NIFTY50-INDEX").strip()
    live_enabled = os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() == "true"
    lots = _positive_int("EMA_LOTS", 1)
    poll_seconds = _positive_int("EMA_POLL_SECONDS", 30)
    max_daily_loss = float(os.getenv("EMA_MAX_LOSS", "5500"))
    if not math.isfinite(max_daily_loss) or max_daily_loss <= 0:
        raise ValueError("EMA_MAX_LOSS must be finite and positive.")
    if os.getenv("EMA_OPTION_EXPIRY_POLICY", "NEXT_WEEK").strip().upper() != "NEXT_WEEK":
        raise RuntimeError("This runner currently supports EMA_OPTION_EXPIRY_POLICY=NEXT_WEEK only.")
    if os.getenv("EMA_OPTION_STRIKE_POLICY", "ATM").strip().upper() != "ATM":
        raise RuntimeError("This runner currently supports EMA_OPTION_STRIKE_POLICY=ATM only.")

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
    completed_trades = 0
    active_date: date | None = None
    last_processed: pd.Timestamp | None = None
    startup_at = datetime.now(IST)
    next_health_at = startup_at + timedelta(hours=1)
    market_data_status = "WAITING FOR FIRST FETCH"
    last_market_data_at: datetime | None = None
    day_started_notified_date: date | None = None
    last_mark_price: float | None = None
    mode = "LIVE" if live_enabled else "PAPER"
    dashboard_state.update_service(
        status="RUNNING",
        mode=mode,
        strategy="EMA",
        underlying=underlying_symbol,
        started_at=startup_at.isoformat(),
        trading_day=startup_at.date().isoformat(),
    )
    dashboard_state.update_market(status=market_data_status, underlying=underlying_symbol)
    dashboard_state.update_account(
        realized_pnl=0.0,
        open_pnl=0.0,
        total_pnl=0.0,
        trade_count=0,
        daily_loss_limit=max_daily_loss,
    )
    dashboard_state.record_event(
        {"at": startup_at.isoformat(), "level": "SUCCESS", "message": "EMA strategy started successfully"}
    )
    notifier.notify(
        _format_strategy_started_alert(
            "LIVE" if live_enabled else "PAPER",
            underlying_symbol,
            lots,
            startup_at,
        )
    )
    LOGGER.info("EMA runner connected to Fyers; mode=%s; lots=%s", "LIVE" if live_enabled else "PAPER", lots)
    LOGGER.info("Option policy: next week's listed expiry, ATM, buy CE/PE by EMA direction.")

    while True:
        now = datetime.now(IST)
        dashboard_state.update_service(
            status="RUNNING",
            mode=mode,
            trading_day=now.date().isoformat(),
            last_loop_at=now.isoformat(),
        )
        if now >= next_health_at:
            notifier.notify(
                _format_health_alert(
                    "LIVE" if live_enabled else "PAPER",
                    position,
                    realized_pnl,
                    market_data_status,
                    last_market_data_at,
                    now,
                )
            )
            next_health_at = now + timedelta(hours=1)
        if active_date != now.date():
            active_date = now.date()
            realized_pnl = 0.0
            completed_trades = 0
            last_processed = None
            day_started_notified_date = None
            last_mark_price = None
            dashboard_state.update_account(
                realized_pnl=0.0,
                open_pnl=0.0,
                total_pnl=0.0,
                trade_count=0,
                daily_loss_limit=max_daily_loss,
            )
        if now.timetz().replace(tzinfo=None) >= day_time(15, 15):
            if position is not None:
                try:
                    closed = _close_position(
                        api,
                        execution,
                        position,
                        live_enabled=live_enabled,
                        product_type=os.getenv("EMA_PRODUCT_TYPE", "INTRADAY"),
                        reason="SQUARE_OFF",
                    )
                    realized_pnl += closed.pnl
                    completed_trades += 1
                    dashboard_state.record_trade(
                        _dashboard_closed_trade(position, closed, "TIME_CUTOFF", mode)
                    )
                    dashboard_state.record_event(
                        {
                            "at": closed.placed_at.isoformat(),
                            "level": "TRADE",
                            "message": "Position closed at session cutoff",
                        }
                    )
                    dashboard_state.set_position(None)
                    notifier.notify(
                        _format_exit_alert(position, closed, "SQUARE_OFF", "LIVE" if live_enabled else "PAPER")
                    )
                except (FyersAPIError, RuntimeError) as exc:
                    LOGGER.critical("Scheduled EMA exit unresolved: %s", exc)
                    raise
            LOGGER.info("EMA session ended at the configured square-off time.")
            dashboard_state.update_service(status="SESSION CLOSED", stopped_at=datetime.now(IST).isoformat())
            dashboard_state.update_account(
                realized_pnl=realized_pnl,
                open_pnl=0.0,
                total_pnl=realized_pnl,
                trade_count=completed_trades,
                daily_loss_limit=max_daily_loss,
            )
            dashboard_state.record_event(
                {"at": datetime.now(IST).isoformat(), "level": "INFO", "message": "EMA session ended at cutoff"}
            )
            notifier.notify(
                _format_eod_alert(
                    realized_pnl,
                    completed_trades,
                    "LIVE" if live_enabled else "PAPER",
                    datetime.now(IST),
                )
            )
            notifier.close()
            return
        try:
            minute_bars = api.history(
                underlying_symbol,
                resolution="1",
                start=now.date() - timedelta(days=7),
                end=now.date(),
            )
            last_market_data_at = datetime.now(IST)
            market_data_status = "OK"
        except FyersAPIError as exc:
            market_data_status = "RETRYING"
            dashboard_state.update_market(status=market_data_status, last_error=type(exc).__name__)
            dashboard_state.record_event(
                {"at": now.isoformat(), "level": "ERROR", "message": "Fyers market-data refresh failed; retrying"}
            )
            LOGGER.error("Fyers market-data refresh failed; retrying: %s", exc)
            _notify_runner_error(
                notifier,
                "Fyers market-data refresh failed; retrying",
                exc,
                "LIVE" if live_enabled else "PAPER",
            )
            time.sleep(poll_seconds)
            continue
        dashboard_state.update_market(
            status="OK",
            underlying=underlying_symbol,
            last_success_at=last_market_data_at.isoformat(),
            last_error=None,
        )

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
        dashboard_state.update_market(
            status=market_data_status,
            underlying=underlying_symbol,
            last_success_at=last_market_data_at.isoformat() if last_market_data_at else None,
            bar_at=current_timestamp.isoformat(),
            spot=_dashboard_number(current.get("close")),
            ema_fast=_dashboard_number(current.get("ema4")),
            ema_mid=_dashboard_number(current.get("ema11")),
            ema_slow=_dashboard_number(current.get("ema18")),
            atr=_dashboard_number(current.get("atr")),
            adx=_dashboard_number(current.get("adx")),
            candles=_dashboard_candles(strategy_frame),
        )
        dashboard_state.update_signal(
            action=decision.action,
            status="SETUP" if decision.action != "HOLD" else "SCANNING",
            at=now.isoformat(),
            stop_spot=_dashboard_number(decision.stop_underlying),
        )
        if decision.action != "HOLD":
            dashboard_state.record_event(
                {"at": now.isoformat(), "level": "SIGNAL", "message": f"EMA {decision.action} signal"}
            )
        if current_timestamp.date() == now.date() and day_started_notified_date != now.date():
            notifier.notify(
                _format_day_started_alert(
                    "LIVE" if live_enabled else "PAPER",
                    now.date(),
                    len(minute_bars),
                    len(bars),
                    last_market_data_at or now,
                )
            )
            day_started_notified_date = now.date()
            dashboard_state.record_event(
                {
                    "at": last_market_data_at.isoformat() if last_market_data_at else now.isoformat(),
                    "level": "SUCCESS",
                    "message": f"Session data ready: {len(minute_bars)} one-minute bars fetched",
                }
            )
        mark_price: float | None = None
        risk_exit_due = False
        if position is not None:
            try:
                mark_price = api.quote_ltp(position.contract.symbol)
                last_mark_price = mark_price
                risk_exit_due = daily_loss_limit_reached(
                    realized_pnl,
                    position.entry_option_price,
                    mark_price,
                    position.quantity,
                    max_daily_loss,
                )
            except FyersAPIError as exc:
                LOGGER.warning("Unable to refresh the open option mark: %s", exc)
                dashboard_state.record_event(
                    {"at": now.isoformat(), "level": "WARN", "message": "Unable to refresh active option mark"}
                )
                _notify_runner_error(
                    notifier,
                    "Unable to refresh the open option mark",
                    exc,
                    "LIVE" if live_enabled else "PAPER",
                )

        if position is not None and (square_off_due or risk_exit_due or decision.action == "EXIT"):
            exit_reason = (
                "SQUARE_OFF"
                if square_off_due
                else "DAILY_MAX_LOSS"
                if risk_exit_due
                else decision.exit_reason
            )
            notifier.notify(
                f"Fyers EMA EXIT signal | mode={'LIVE' if live_enabled else 'PAPER'}\n"
                f"Symbol: {position.contract.symbol}\n"
                f"Reason: {exit_reason}\n"
                f"Underlying: {float(current['close']):.2f}\n"
                f"This is a signal, not order/fill confirmation."
            )
            try:
                closed = _close_position(
                    api,
                    execution,
                    position,
                    live_enabled=live_enabled,
                    product_type=os.getenv("EMA_PRODUCT_TYPE", "INTRADAY"),
                    reason=exit_reason,
                    mark_price=mark_price,
                )
                realized_pnl += closed.pnl
                completed_trades += 1
                dashboard_state.record_trade(
                    _dashboard_closed_trade(position, closed, exit_reason, mode)
                )
                dashboard_state.record_event(
                    {
                        "at": closed.placed_at.isoformat(),
                        "level": "TRADE",
                        "message": f"EMA position closed: {exit_reason}",
                    }
                )
                last_mark_price = None
                LOGGER.info(
                    "EMA exit confirmed | symbol=%s | reason=%s | pnl=%.2f | daily=%.2f",
                    position.contract.symbol,
                    exit_reason,
                    closed.pnl,
                    realized_pnl,
                )
                notifier.notify(
                    _format_exit_alert(position, closed, exit_reason, "LIVE" if live_enabled else "PAPER")
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
                entry_placed_at = datetime.now(IST)
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
                        rejection = RuntimeError(result.reason or "Fyers rejected the EMA entry order.")
                        _notify_runner_error(notifier, "Fyers rejected EMA entry", rejection, "LIVE")
                        dashboard_state.record_event(
                            {
                                "at": datetime.now(IST).isoformat(),
                                "level": "ERROR",
                                "message": "Fyers rejected EMA entry",
                            }
                        )
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
                        entry_placed_at,
                    )
                    last_mark_price = entry_price
                    dashboard_state.set_position(_dashboard_position(position, last_mark_price))
                    dashboard_state.record_event(
                        {
                            "at": entry_placed_at.isoformat(),
                            "level": "TRADE",
                            "message": f"EMA {direction} position opened",
                        }
                    )
                    notifier.notify(
                        _format_entry_alert(position, lots, "LIVE" if live_enabled else "PAPER")
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

        if position is None:
            dashboard_state.set_position(None)
            dashboard_state.update_account(
                realized_pnl=realized_pnl,
                open_pnl=0.0,
                total_pnl=realized_pnl,
                trade_count=completed_trades,
                daily_loss_limit=max_daily_loss,
            )
        else:
            position_snapshot = _dashboard_position(position, last_mark_price)
            open_pnl = position_snapshot["open_pnl"] if position_snapshot is not None else None
            dashboard_state.set_position(position_snapshot)
            dashboard_state.update_account(
                realized_pnl=realized_pnl,
                open_pnl=open_pnl,
                total_pnl=realized_pnl + float(open_pnl or 0.0),
                trade_count=completed_trades,
                daily_loss_limit=max_daily_loss,
            )
        if current_timestamp != last_processed:
            last_processed = current_timestamp
        time.sleep(poll_seconds)
