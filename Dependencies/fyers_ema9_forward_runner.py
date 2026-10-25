"""Forward-test the EMA9 + 2-step ITM strategy on live Fyers data."""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import math
import os
import sys
import tempfile
import time as sleep_time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import talib
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Dependencies.broker_contract import OrderStatus  # noqa: E402
from Dependencies.fyers_api import FyersAPI, FyersAPIError  # noqa: E402
from Dependencies.fyers_execution import FyersExecutionClient  # noqa: E402
from Dependencies.fyers_dashboard_state import FyersDashboardState  # noqa: E402
from Dependencies.fyers_nifty_ema9_touch_backtest import run_dashboard_backtest  # noqa: E402
from Dependencies.fyers_option_backtest import _completed_five_minute_bars  # noqa: E402
from Dependencies.fyers_options import (  # noqa: E402
    FyersOptionContract,
    select_atm_option,
    select_nearest_expiry,
    select_next_week_expiry,
)
from Dependencies.secret_redaction import environment_secrets, redact_text  # noqa: E402
from Dependencies.fyers_telegram_auth import (  # noqa: E402
    ENV_FILE,
    FyersTelegramAuthService,
    load_persisted_access_token,
)
from Dependencies.telegram_notifier import TelegramNotifier  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
ENTRY_START = time(9, 25)
MARKET_OPEN = time(9, 15)
SQUARE_OFF = time(15, 15)
MARKET_DATA_END = time(15, 20)
HEALTH_ALERT_INTERVAL = timedelta(hours=1)
EMA9_BUFFER_POINTS = 3.0
DAILY_OPTION_LOSS_CAP = 3500.0
OPTION_STRIKE_STEPS = 2
LOGGER = logging.getLogger("fyers_ema9_forward")
STATE_VERSION = 1
TRADE_COLUMNS = [
    "trade_id",
    "mode",
    "instrument",
    "signal_bar_start",
    "entry_time",
    "exit_time",
    "exit_trigger_time",
    "signal",
    "exit_reason",
    "option_symbol",
    "option_expiry",
    "option_type",
    "strike",
    "lot_size",
    "quantity",
    "entry_index",
    "exit_index",
    "entry_option_price",
    "exit_option_price",
    "option_pnl",
    "daily_option_pnl",
    "daily_option_loss_cap",
]


@dataclass(frozen=True, slots=True)
class InstrumentConfig:
    name: str
    underlying: str
    index_symbol: str
    exchange: str
    expiry_policy: str

    def __post_init__(self) -> None:
        if self.exchange not in {"NSE", "BSE"}:
            raise ValueError("Instrument exchange must be NSE or BSE.")
        if self.expiry_policy not in {"NEXT_ISO_WEEK", "NEAREST"}:
            raise ValueError("Unsupported option expiry policy.")


@dataclass(slots=True)
class ForwardPosition:
    trade_id: str
    signal: str
    signal_bar_start: str
    entry_time: str
    entry_index: float
    trailing_stop: float
    contract: FyersOptionContract
    quantity: int
    entry_option_price: float


def execution_mode_from_environment(environment: dict[str, str]) -> str:
    mode = environment.get("EMA9_EXECUTION_MODE", "PAPER").strip().upper()
    if mode not in {"PAPER", "LIVE"}:
        raise ValueError("EMA9_EXECUTION_MODE must be PAPER or LIVE.")
    return mode


def _is_market_data_session(at: datetime) -> bool:
    local_at = at.astimezone(IST)
    return (
        local_at.weekday() < 5
        and MARKET_OPEN <= local_at.time() < MARKET_DATA_END
    )


def select_itm_contract(
    rows: list[dict[str, Any]],
    *,
    action: str,
    spot: float,
    trade_day: date,
    config: InstrumentConfig,
    strike_steps: int = OPTION_STRIKE_STEPS,
) -> FyersOptionContract:
    if strike_steps <= 0:
        raise ValueError("strike_steps must be positive.")
    instrument_rows: list[dict[str, Any]] = []
    expiries: set[date] = set()
    for row in rows:
        if str(row.get("underlying", "")).strip().upper() != config.underlying:
            continue
        if str(row.get("option_type", "")).strip().upper() not in {"CE", "PE"}:
            continue
        try:
            expiry_value = row["expiry"]
            if isinstance(expiry_value, (int, float)):
                expiry = datetime.fromtimestamp(float(expiry_value), tz=UTC).date()
            else:
                expiry = date.fromisoformat(str(expiry_value)[:10])
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            continue
        normalized = {**row, "expiry": expiry}
        instrument_rows.append(normalized)
        expiries.add(expiry)

    if config.expiry_policy == "NEXT_ISO_WEEK":
        expiry = select_next_week_expiry(expiries, today=trade_day)
    else:
        expiry = select_nearest_expiry(expiries, today=trade_day)
    expiry_rows = [row for row in instrument_rows if row["expiry"] == expiry]
    atm = select_atm_option(expiry_rows, action=action, spot=spot, expiry=expiry)
    strikes_set: set[float] = set()
    for row in expiry_rows:
        if str(row.get("option_type", "")).upper() != atm.option_type:
            continue
        try:
            strike = float(row["strike_price"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(strike) and strike > 0:
            strikes_set.add(strike)
    strikes = sorted(strikes_set)
    atm_index = strikes.index(atm.strike)
    target_index = atm_index + (-strike_steps if atm.option_type == "CE" else strike_steps)
    if not 0 <= target_index < len(strikes):
        raise ValueError(
            f"No {strike_steps}-step ITM {atm.option_type} contract is listed for {expiry}."
        )
    target_strike = strikes[target_index]
    return select_atm_option(
        [
            row
            for row in expiry_rows
            if str(row.get("option_type", "")).upper() == atm.option_type
            and float(row["strike_price"]) == target_strike
        ],
        action=action,
        spot=spot,
        expiry=expiry,
    )


def _load_signal_logic() -> Any:
    path = ROOT_DIR / "Signal Generators" / "ema_trend_strategy_logic.py"
    spec = importlib.util.spec_from_file_location("ema_trend_strategy_logic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load EMA signal logic from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _localize_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize(IST) if timestamp.tzinfo is None else timestamp.tz_convert(IST)


def _format_entry_alert(
    position: ForwardPosition, *, instrument: str, mode: str
) -> str:
    direction = "LONG" if position.signal == "ENTER_LONG" else "SHORT"
    contract = position.contract
    placed_at = _localize_timestamp(position.entry_time)
    return (
        "🚀 ENTRY · EMA\n"
        f"Direction: {direction}\n"
        f"Instrument: BUY {instrument}-{contract.expiry:%b%Y}-{contract.strike:g}-{contract.option_type}\n"
        f"  Entry: ₹{position.entry_option_price:.2f}\n"
        f"Qty: {position.quantity} (1 lot(s) x {contract.lot_size})\n"
        f"Index entry: {position.entry_index:.2f}\n"
        f"[{mode}] {placed_at:%Y-%m-%d %H:%M:%S} IST"
    )


def _format_exit_alert(
    position: ForwardPosition,
    trade: dict[str, Any],
    *,
    instrument: str,
    mode: str,
) -> str:
    pnl = float(trade["option_pnl"])
    direction = "LONG" if position.signal == "ENTER_LONG" else "SHORT"
    if trade["exit_reason"] == "SQUARE_OFF":
        display_reason = "TIME_CUTOFF"
    elif trade["exit_reason"] == "OPTION_DAILY_LOSS_CAP":
        display_reason = f"OPTION_DAILY_LOSS_CAP · ₹{float(trade['daily_option_loss_cap']):g}"
    else:
        display_reason = str(trade["exit_reason"])
    exit_time = _localize_timestamp(trade["exit_time"])
    return (
        f"🏁 EXIT · EMA ({display_reason})\n"
        f"Direction: {direction}\n"
        f"Instrument: SELL {instrument}-{position.contract.expiry:%b%Y}-"
        f"{position.contract.strike:g}-{position.contract.option_type}\n"
        f"  Entry: ₹{float(trade['entry_option_price']):.2f} → "
        f"Exit: ₹{float(trade['exit_option_price']):.2f}\n"
        f"Qty: {position.quantity} (lot size {position.contract.lot_size})\n"
        f"⚪ P&L: {'+₹' if pnl >= 0 else '-₹'}{abs(pnl):.2f}\n"
        f"Daily option P&L: ₹{float(trade['daily_option_pnl']):+.2f}\n"
        f"[{mode}] {exit_time:%Y-%m-%d %H:%M:%S} IST"
    )


def _format_eod_alert(
    *,
    instrument: str,
    mode: str,
    at: datetime,
    trade_count: int,
    realized_pnl: float,
    loss_cap: float,
    halted: bool,
) -> str:
    at = at.astimezone(IST)
    return (
        f"📊 End-of-Day P&L [{mode}]\n"
        f"{at:%Y-%m-%d %H:%M:%S} IST\n"
        f"{instrument} [{mode}]: {'+₹' if realized_pnl >= 0 else '-₹'}"
        f"{abs(realized_pnl):.2f} ({trade_count} trade(s))\n"
        "──────────\n"
        f"Total: {'+₹' if realized_pnl >= 0 else '-₹'}{abs(realized_pnl):.2f} "
        f"({trade_count} trade(s))\n"
        f"Daily loss cap: ₹{loss_cap:.2f} | "
        f"{'HALTED' if halted else 'not halted'}"
    )


def _format_health_alert(
    runner: EMA9ForwardRunner,
    *,
    market_data_status: str,
    last_market_data_at: datetime | None,
    at: datetime,
) -> str:
    position = runner.position
    if position is None:
        position_status = "FLAT"
    else:
        direction = "LONG" if position.signal == "ENTER_LONG" else "SHORT"
        position_status = (
            f"{direction} {position.contract.symbol} x {position.quantity}"
        )
    last_success = (
        f"{_localize_timestamp(last_market_data_at):%Y-%m-%d %H:%M:%S} IST"
        if last_market_data_at is not None
        else "not received yet"
    )
    realized_pnl = runner.daily_realized_option_pnl
    return (
        "💓 HEALTH · EMA9\n"
        "Status: RUNNING\n"
        f"Instrument: {runner.config.name}\n"
        f"Mode: {runner.mode}\n"
        f"Time: {_localize_timestamp(at):%Y-%m-%d %H:%M:%S} IST\n"
        f"Market data: {market_data_status} (last success: {last_success})\n"
        f"Position: {position_status}\n"
        f"Realized option P&L today: {'+₹' if realized_pnl >= 0 else '-₹'}"
        f"{abs(realized_pnl):.2f} ({runner.daily_trade_count} trade(s))\n"
        f"Daily loss cap: ₹{runner.daily_loss_cap:.2f} | "
        f"{'HALTED' if runner.daily_loss_halted else 'not halted'}"
    )


def _format_error_alert(
    context: str, error: Exception, *, instrument: str, mode: str
) -> str:
    details = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    details = redact_text(details, environment_secrets(os.environ)).strip()
    if len(details) > 2400:
        details = "[Earlier traceback details truncated]\n" + details[-2400:]
    return (
        f"🚨 ERROR · EMA\n"
        f"Instrument: {instrument}\n"
        f"Mode: {mode}\n"
        f"Context: {context}\n"
        f"Time: {datetime.now(IST):%Y-%m-%d %H:%M:%S} IST\n\n"
        f"{details}"
    )


def _notify_runner_error(
    notifier: TelegramNotifier,
    context: str,
    error: Exception,
    *,
    instrument: str,
    mode: str,
) -> None:
    notifier.notify_error(
        f"{instrument}:{context}:{type(error).__name__}",
        _format_error_alert(context, error, instrument=instrument, mode=mode),
    )


def _create_telegram_notifier(environment: dict[str, str]) -> TelegramNotifier:
    bot_token = environment.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = (
        environment.get("TELEGRAM_CHAT_ID", "").strip()
        or environment.get("FYERS_AUTH_TELEGRAM_CHAT_ID", "").strip()
    )
    if not bot_token or not chat_id:
        raise ValueError(
            "EMA9 Telegram alerts require TELEGRAM_BOT_TOKEN and either TELEGRAM_CHAT_ID "
            "or FYERS_AUTH_TELEGRAM_CHAT_ID."
        )
    return TelegramNotifier(bot_token, chat_id, enabled=True)


def _quote_ltp(api: FyersAPI, symbol: str, *, now: datetime) -> float:
    response = api.quotes([symbol])
    entries = response.get("d")
    if not isinstance(entries, list):
        raise FyersAPIError("Fyers quote response did not contain quote data.")
    for entry in entries:
        if not isinstance(entry, dict) or str(entry.get("n", "")).strip() != symbol:
            continue
        values = entry.get("v")
        if not isinstance(values, dict):
            break
        try:
            price = float(values.get("lp"))
            timestamp_value = values.get("tt", values.get("timestamp"))
            quote_timestamp = float(timestamp_value)
        except (TypeError, ValueError, OverflowError):
            break
        if not math.isfinite(quote_timestamp) or quote_timestamp <= 0:
            break
        if quote_timestamp > 10**12:
            quote_timestamp /= 1000
        try:
            quoted_at = datetime.fromtimestamp(quote_timestamp, tz=IST)
        except (OverflowError, OSError, ValueError):
            break
        age = (now.astimezone(IST) - quoted_at).total_seconds()
        if not math.isfinite(price) or price <= 0 or age < -30 or age > 60:
            break
        return price
    raise FyersAPIError(f"Fyers did not return a fresh, valid quote for {symbol}.")


class EMA9ForwardRunner:
    """Stateful strategy runner shared by the isolated instrument entry points."""

    def __init__(
        self,
        *,
        api: FyersAPI,
        execution: FyersExecutionClient,
        config: InstrumentConfig,
        mode: str,
        state_path: Path,
        trades_path: Path,
        notifier: Any | None = None,
        dashboard_state: FyersDashboardState | None = None,
        signal_logic: Any,
        strategy_config: Any,
        daily_loss_cap: float = DAILY_OPTION_LOSS_CAP,
        ema9_buffer_points: float = EMA9_BUFFER_POINTS,
    ) -> None:
        if mode not in {"PAPER", "LIVE"}:
            raise ValueError("mode must be PAPER or LIVE.")
        if not math.isfinite(daily_loss_cap) or daily_loss_cap <= 0:
            raise ValueError("daily_loss_cap must be finite and positive.")
        if not math.isfinite(ema9_buffer_points) or ema9_buffer_points < 0:
            raise ValueError("ema9_buffer_points must be finite and non-negative.")
        self.api = api
        self.execution = execution
        self.config = config
        self.mode = mode
        self.state_path = state_path
        self.trades_path = trades_path
        self.notifier = notifier
        self.dashboard_state = dashboard_state
        self.signal_logic = signal_logic
        self.strategy_config = strategy_config
        self.daily_loss_cap = daily_loss_cap
        self.ema9_buffer_points = ema9_buffer_points
        self.position: ForwardPosition | None = None
        self.trading_day: date | None = None
        self.daily_realized_option_pnl = 0.0
        self.daily_trade_count = 0
        self.daily_loss_halted = False
        self.eod_report_sent = False
        self.last_processed_minute: pd.Timestamp | None = None
        self.last_processed_bar: pd.Timestamp | None = None
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("version") != STATE_VERSION or state.get("instrument") != self.config.name:
                raise RuntimeError("Forward-runner state is incompatible with this instrument.")
            if state.get("mode") != self.mode:
                raise RuntimeError("Saved runner state mode differs from EMA9_EXECUTION_MODE.")
            if float(state["daily_loss_cap"]) != self.daily_loss_cap:
                raise RuntimeError("Saved runner state uses a different daily loss cap.")
            self.trading_day = (
                date.fromisoformat(state["trading_day"]) if state.get("trading_day") else None
            )
            self.daily_realized_option_pnl = float(state["daily_realized_option_pnl"])
            self.daily_trade_count = int(state.get("daily_trade_count", 0))
            self.daily_loss_halted = bool(state["daily_loss_halted"])
            self.eod_report_sent = bool(state.get("eod_report_sent", False))
            minute = state.get("last_processed_minute")
            bar = state.get("last_processed_bar")
            self.last_processed_minute = _localize_timestamp(minute) if minute else None
            self.last_processed_bar = _localize_timestamp(bar) if bar else None
            raw_position = state.get("position")
            if raw_position:
                contract_data = raw_position.pop("contract")
                contract = FyersOptionContract(
                    symbol=contract_data["symbol"],
                    expiry=date.fromisoformat(contract_data["expiry"]),
                    option_type=contract_data["option_type"],
                    strike=float(contract_data["strike"]),
                    lot_size=int(contract_data["lot_size"]),
                )
                self.position = ForwardPosition(contract=contract, **raw_position)
        except RuntimeError:
            raise
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to load forward-runner state from {self.state_path}.") from exc
        if not math.isfinite(self.daily_realized_option_pnl):
            raise RuntimeError("Forward-runner state contains invalid realized P&L.")

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": STATE_VERSION,
            "instrument": self.config.name,
            "mode": self.mode,
            "daily_loss_cap": self.daily_loss_cap,
            "ema9_buffer_points": self.ema9_buffer_points,
            "trading_day": self.trading_day.isoformat() if self.trading_day else None,
            "daily_realized_option_pnl": self.daily_realized_option_pnl,
            "daily_trade_count": self.daily_trade_count,
            "daily_loss_halted": self.daily_loss_halted,
            "eod_report_sent": self.eod_report_sent,
            "last_processed_minute": (
                self.last_processed_minute.isoformat() if self.last_processed_minute is not None else None
            ),
            "last_processed_bar": (
                self.last_processed_bar.isoformat() if self.last_processed_bar is not None else None
            ),
            "position": (
                {
                    **{
                        key: value
                        for key, value in asdict(self.position).items()
                        if key != "contract"
                    },
                    "contract": {
                        **asdict(self.position.contract),
                        "expiry": self.position.contract.expiry.isoformat(),
                    },
                }
                if self.position is not None
                else None
            ),
        }
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.state_path.name}.", suffix=".tmp", dir=self.state_path.parent
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
                json.dump(state, temporary_file, indent=2)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_name, self.state_path)
        except OSError:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def _append_trade(self, row: dict[str, Any]) -> None:
        self.trades_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.trades_path.exists()
        with self.trades_path.open("a", newline="", encoding="utf-8") as trades_file:
            writer = csv.DictWriter(trades_file, fieldnames=TRADE_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _begin_day(self, trading_day: date) -> None:
        if self.trading_day == trading_day:
            return
        if self.position is not None:
            raise RuntimeError("An open position cannot carry into a new trading day.")
        self.trading_day = trading_day
        self.daily_realized_option_pnl = 0.0
        self.daily_trade_count = 0
        self.daily_loss_halted = False
        self.eod_report_sent = False
        self.last_processed_minute = None
        self.last_processed_bar = None
        self._save_state()

    def _place_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        tag: str,
        fallback_price: float,
    ) -> float:
        result = self.execution.place_market_order(
            symbol,
            side,
            quantity,
            exchange_segment=f"{self.config.exchange}_FNO",
            product_type="INTRADAY",
            order_tag=tag,
        )
        if result.status is not OrderStatus.FILLED:
            raise RuntimeError(
                f"Fyers {side.lower()} order was not fully confirmed ({result.status.value}); "
                "halt and reconcile the account."
            )
        if result.average_fill_price <= 0:
            LOGGER.warning(
                "Fyers confirmed %s %s without an average fill price; recording the pre-order quote.",
                side,
                symbol,
            )
            return fallback_price
        return result.average_fill_price

    def _close_position(
        self,
        *,
        at: datetime,
        exit_index: float,
        reason: str,
        quote_price: float | None = None,
        trigger_time: datetime | None = None,
    ) -> None:
        position = self.position
        if position is None:
            raise RuntimeError("Cannot close an absent EMA9 position.")
        if quote_price is not None:
            fallback_price = quote_price
        elif self.mode == "LIVE":
            try:
                fallback_price = _quote_ltp(self.api, position.contract.symbol, now=at)
            except FyersAPIError:
                fallback_price = position.entry_option_price
        else:
            fallback_price = _quote_ltp(self.api, position.contract.symbol, now=at)
        if self.mode == "LIVE":
            exit_price = self._place_order(
                symbol=position.contract.symbol,
                side="SELL",
                quantity=position.quantity,
                tag="EMA9-EXIT",
                fallback_price=fallback_price,
            )
        else:
            exit_price = fallback_price
        pnl = (exit_price - position.entry_option_price) * position.quantity
        self.daily_realized_option_pnl += pnl
        self.daily_trade_count += 1
        self.daily_loss_halted = (
            self.daily_loss_halted
            or self.daily_realized_option_pnl <= -self.daily_loss_cap
        )
        trade = {
            "trade_id": position.trade_id,
            "mode": self.mode,
            "instrument": self.config.name,
            "signal_bar_start": position.signal_bar_start,
            "entry_time": position.entry_time,
            "exit_time": at.isoformat(),
            "exit_trigger_time": (trigger_time or at).isoformat(),
            "signal": position.signal,
            "exit_reason": reason,
            "option_symbol": position.contract.symbol,
            "option_expiry": position.contract.expiry.isoformat(),
            "option_type": position.contract.option_type,
            "strike": position.contract.strike,
            "lot_size": position.contract.lot_size,
            "quantity": position.quantity,
            "entry_index": position.entry_index,
            "exit_index": exit_index,
            "entry_option_price": position.entry_option_price,
            "exit_option_price": exit_price,
            "option_pnl": pnl,
            "daily_option_pnl": self.daily_realized_option_pnl,
            "daily_option_loss_cap": self.daily_loss_cap,
        }
        self._append_trade(trade)
        self.position = None
        self._save_state()
        LOGGER.info(
            "%s EXIT %s reason=%s option=%.2f P&L=%.2f daily=%.2f",
            self.config.name,
            position.contract.symbol,
            reason,
            exit_price,
            pnl,
            self.daily_realized_option_pnl,
        )
        self._notify(
            _format_exit_alert(
                position,
                trade,
                instrument=self.config.name,
                mode=self.mode,
            )
        )
        if self.dashboard_state is not None:
            self.dashboard_state.set_position(None)
            self.dashboard_state.record_trade(
                {
                    "exit_time": trade["exit_time"],
                    "direction": direction,
                    "display_symbol": position.contract.symbol,
                    "symbol": position.contract.symbol,
                    "quantity": position.quantity,
                    "entry_price": trade["entry_option_price"],
                    "exit_price": trade["exit_option_price"],
                    "pnl": pnl,
                    "reason": reason,
                }
            )
            self.dashboard_state.update_account(
                realized_pnl=self.daily_realized_option_pnl,
                total_pnl=self.daily_realized_option_pnl,
                trade_count=self.daily_trade_count,
                daily_loss_limit=self.daily_loss_cap,
                open_pnl=0.0,
            )

    def _open_position(
        self,
        *,
        action: str,
        bar_start: pd.Timestamp,
        bar_close: datetime,
        quote_at: datetime,
        entry_index: float,
        entry_ema9: float,
    ) -> None:
        master_rows = self.api.option_master(self.config.exchange)
        contract = select_itm_contract(
            master_rows,
            action=action,
            spot=entry_index,
            trade_day=bar_close.date(),
            config=self.config,
        )
        quote = _quote_ltp(self.api, contract.symbol, now=quote_at)
        quantity = contract.lot_size
        entry_price = quote
        if self.mode == "LIVE":
            entry_price = self._place_order(
                symbol=contract.symbol,
                side="BUY",
                quantity=quantity,
                tag="EMA9-ENTRY",
                fallback_price=quote,
            )
        direction = 1.0 if action == "ENTER_LONG" else -1.0
        position = ForwardPosition(
            trade_id=uuid.uuid4().hex,
            signal=action,
            signal_bar_start=bar_start.isoformat(),
            entry_time=quote_at.isoformat(),
            entry_index=entry_index,
            trailing_stop=entry_ema9 - direction * self.ema9_buffer_points,
            contract=contract,
            quantity=quantity,
            entry_option_price=entry_price,
        )
        self.position = position
        self._save_state()
        self._update_dashboard_position(mark_price=entry_price, open_pnl=0.0)
        self._notify(
            _format_entry_alert(
                position,
                instrument=self.config.name,
                mode=self.mode,
            )
        )
        LOGGER.info(
            "%s %s ENTRY %s expiry=%s strike=%s qty=%s option=%.2f index=%.2f stop=%.2f",
            self.mode,
            self.config.name,
            action,
            contract.expiry,
            contract.strike,
            quantity,
            entry_price,
            entry_index,
            position.trailing_stop,
        )

    def _enriched_bars(self, minute_bars: pd.DataFrame) -> pd.DataFrame:
        bars = _completed_five_minute_bars(minute_bars)
        if bars.empty:
            return bars
        enriched = self.signal_logic.build_ema_trend_with_indicators(
            bars, self.strategy_config
        ).copy()
        enriched["ema9"] = talib.EMA(
            pd.to_numeric(enriched["close"], errors="coerce").to_numpy(dtype=float),
            timeperiod=9,
        )
        enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], errors="coerce")
        if enriched["timestamp"].dt.tz is None:
            enriched["timestamp"] = enriched["timestamp"].dt.tz_localize(IST)
        else:
            enriched["timestamp"] = enriched["timestamp"].dt.tz_convert(IST)
        return enriched.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    def check_open_position_risk(self, *, now: datetime) -> None:
        now = now.astimezone(IST)
        position = self.position
        if position is None:
            self._update_dashboard_position()
            return
        if now.time() >= SQUARE_OFF:
            self._close_position(
                at=now,
                exit_index=position.entry_index,
                reason="SQUARE_OFF",
            )
            return
        try:
            mark = _quote_ltp(self.api, position.contract.symbol, now=now)
        except FyersAPIError:
            if self.mode != "LIVE":
                raise
            self._close_position(
                at=now,
                exit_index=position.entry_index,
                reason="STALE_OPTION_QUOTE",
                quote_price=position.entry_option_price,
            )
            return
        open_pnl = (mark - position.entry_option_price) * position.quantity
        self._update_dashboard_position(mark_price=mark, open_pnl=open_pnl)
        if self.daily_realized_option_pnl + open_pnl <= -self.daily_loss_cap:
            self.daily_loss_halted = True
            self._save_state()
            self._close_position(
                at=now,
                exit_index=position.entry_index,
                reason="OPTION_DAILY_LOSS_CAP",
                quote_price=mark,
            )

    def _update_dashboard_position(
        self,
        *,
        mark_price: float | None = None,
        open_pnl: float | None = None,
    ) -> None:
        if self.dashboard_state is None:
            return
        position = self.position
        if position is None:
            displayed_open_pnl = 0.0
            self.dashboard_state.set_position(None)
        else:
            displayed_mark = (
                position.entry_option_price if mark_price is None else mark_price
            )
            displayed_open_pnl = (
                (displayed_mark - position.entry_option_price) * position.quantity
                if open_pnl is None
                else open_pnl
            )
            self.dashboard_state.set_position(
                {
                    "direction": "LONG" if position.signal == "ENTER_LONG" else "SHORT",
                    "quantity": position.quantity,
                    "symbol": position.contract.symbol,
                    "display_symbol": position.contract.symbol,
                    "entry_price": position.entry_option_price,
                    "mark_price": displayed_mark,
                    "entry_spot": position.entry_index,
                    "stop_spot": position.trailing_stop,
                    "opened_at": position.entry_time,
                    "open_pnl": displayed_open_pnl,
                }
            )
        self.dashboard_state.update_account(
            realized_pnl=self.daily_realized_option_pnl,
            open_pnl=displayed_open_pnl,
            total_pnl=self.daily_realized_option_pnl + displayed_open_pnl,
            trade_count=self.daily_trade_count,
            daily_loss_limit=self.daily_loss_cap,
        )

    def send_end_of_day_report(self, *, now: datetime) -> None:
        now = now.astimezone(IST)
        if (
            now.time() < SQUARE_OFF
            or self.trading_day != now.date()
            or self.eod_report_sent
        ):
            return
        self.eod_report_sent = True
        self._save_state()
        self._notify(
            _format_eod_alert(
                instrument=self.config.name,
                mode=self.mode,
                at=now,
                trade_count=self.daily_trade_count,
                realized_pnl=self.daily_realized_option_pnl,
                loss_cap=self.daily_loss_cap,
                halted=self.daily_loss_halted,
            )
        )

    def _notify(self, message: str) -> None:
        if self.notifier is None:
            return
        if not self.notifier.notify(message):
            LOGGER.warning("Telegram EMA9 alert was not queued.")

    def process_market_data(self, minute_bars: pd.DataFrame, *, now: datetime) -> None:
        now = now.astimezone(IST)
        self._begin_day(now.date())
        if self.dashboard_state is not None:
            self.dashboard_state.update_service(
                status="RUNNING",
                mode=self.mode,
                strategy="EMA9",
                instrument=self.config.name,
                underlying=self.config.index_symbol,
                trading_day=now.date().isoformat(),
                last_loop_at=now.isoformat(),
            )
            self.dashboard_state.update_account(
                realized_pnl=self.daily_realized_option_pnl,
                total_pnl=self.daily_realized_option_pnl,
                trade_count=self.daily_trade_count,
                daily_loss_limit=self.daily_loss_cap,
            )
        self.check_open_position_risk(now=now)
        self.send_end_of_day_report(now=now)
        if minute_bars.empty:
            return
        minutes = minute_bars.copy()
        minutes["timestamp"] = pd.to_datetime(minutes["timestamp"], errors="coerce")
        if minutes["timestamp"].dt.tz is None:
            minutes["timestamp"] = minutes["timestamp"].dt.tz_localize(IST)
        else:
            minutes["timestamp"] = minutes["timestamp"].dt.tz_convert(IST)
        minutes = minutes.dropna(subset=["timestamp"]).sort_values("timestamp")
        minutes = minutes.drop_duplicates("timestamp", keep="last")
        minutes = minutes[
            (minutes["timestamp"].dt.time >= MARKET_OPEN)
            & (minutes["timestamp"] + pd.Timedelta(minutes=1) <= pd.Timestamp(now))
        ]
        enriched = self._enriched_bars(minutes)
        if enriched.empty:
            return
        if self.dashboard_state is not None:
            latest = enriched.iloc[-1]
            def dashboard_number(column: str) -> float | None:
                value = latest.get(column)
                if value is None or pd.isna(value):
                    return None
                number = float(value)
                return number if math.isfinite(number) else None

            chart_rows = []
            for bar in enriched.tail(300).itertuples(index=False):
                bar_time = _localize_timestamp(bar.timestamp)
                def chart_number(column: str) -> float | None:
                    value = getattr(bar, column, None)
                    if value is None or pd.isna(value):
                        return None
                    number = float(value)
                    return number if math.isfinite(number) else None

                chart_rows.append(
                    {
                        "time": int(bar_time.timestamp()),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "ema_fast": chart_number("ema_fast"),
                        "ema_mid": chart_number("ema_mid"),
                        "ema_slow": chart_number("ema_slow"),
                    }
                )
            self.dashboard_state.update_market(
                status="OK",
                underlying=self.config.index_symbol,
                last_success_at=now.isoformat(),
                bar_at=_localize_timestamp(latest["timestamp"]).isoformat(),
                spot=dashboard_number("close"),
                ema_fast=dashboard_number("ema_fast"),
                ema_mid=dashboard_number("ema_mid"),
                ema_slow=dashboard_number("ema_slow"),
                atr=dashboard_number("atr"),
                adx=dashboard_number("adx"),
                candles=chart_rows,
            )
        today_minutes = minutes[minutes["timestamp"].dt.date == now.date()]

        bar_starts = pd.DatetimeIndex(enriched["timestamp"])
        bar_lookup = {timestamp: index for index, timestamp in enumerate(bar_starts)}
        position_open_at_start = self.position is not None
        for minute in today_minutes.itertuples(index=False):
            minute_start = _localize_timestamp(minute.timestamp)
            if (
                self.last_processed_minute is not None
                and minute_start <= self.last_processed_minute
            ):
                continue
            minute_close = minute_start + pd.Timedelta(minutes=1)
            bucket_start = minute_start.floor("5min")
            bar_index = bar_lookup.get(bucket_start)
            if (
                self.position is not None
                and minute_start >= _localize_timestamp(self.position.entry_time)
            ):
                prior_index = bar_lookup.get(
                    bucket_start - pd.Timedelta(minutes=5), -1
                )
                prior_ema9 = (
                    float(enriched.iloc[prior_index]["ema9"])
                    if prior_index >= 0
                    else math.nan
                )
                current_bar_is_closed = (
                    bar_index is not None
                    and minute_close == bucket_start + pd.Timedelta(minutes=5)
                )
                if current_bar_is_closed:
                    current_bar = enriched.iloc[bar_index]
                    stop = self.position.trailing_stop
                    direction = 1.0 if self.position.signal == "ENTER_LONG" else -1.0
                    stop_hit = (
                        float(current_bar["close"]) <= stop
                        if direction > 0
                        else float(current_bar["close"]) >= stop
                    )
                    if stop_hit or minute_close.time() >= SQUARE_OFF:
                        reason = "TRAILING_STOP" if stop_hit else "SQUARE_OFF"
                        self._close_position(
                            at=now,
                            exit_index=float(current_bar["close"]),
                            reason=reason,
                            trigger_time=minute_close.to_pydatetime(),
                        )
                    else:
                        current_ema9 = float(current_bar["ema9"])
                        if math.isfinite(current_ema9):
                            candidate = current_ema9 - direction * self.ema9_buffer_points
                            self.position.trailing_stop = (
                                max(self.position.trailing_stop, candidate)
                                if direction > 0
                                else min(self.position.trailing_stop, candidate)
                            )
                if self.position is not None and math.isfinite(prior_ema9):
                    direction = 1.0 if self.position.signal == "ENTER_LONG" else -1.0
                    prior_stop = prior_ema9 - direction * self.ema9_buffer_points
                    touched = (
                        float(minute.low) <= prior_stop
                        if direction > 0
                        else float(minute.high) >= prior_stop
                    )
                    if touched:
                        exit_index = (
                            float(minute.open)
                            if (
                                float(minute.open) <= prior_stop
                                if direction > 0
                                else float(minute.open) >= prior_stop
                            )
                            else prior_stop
                        )
                        self._close_position(
                            at=now,
                            exit_index=exit_index,
                            reason="EMA9_TOUCH",
                            trigger_time=minute_close.to_pydatetime(),
                        )
            self.last_processed_minute = minute_start
            self._save_state()

        if self.position is not None or self.daily_loss_halted:
            return
        latest_index = len(enriched) - 1
        latest = enriched.iloc[latest_index]
        bar_start = _localize_timestamp(latest["timestamp"])
        bar_close = bar_start + pd.Timedelta(minutes=5)
        if self.last_processed_bar is not None and bar_start <= self.last_processed_bar:
            return
        age = (pd.Timestamp(now) - bar_close).total_seconds()
        if (
            bar_close.date() != now.date()
            or bar_close.time() < ENTRY_START
            or bar_close.time() >= SQUARE_OFF
            or age < 0
            or age > 120
            or position_open_at_start
        ):
            self.last_processed_bar = bar_start
            self._save_state()
            return
        engine = self.signal_logic.EMATrendSignalEngine(self.strategy_config)
        decision = engine.evaluate_candle(enriched.iloc[: latest_index + 1])
        if decision.action not in {"ENTER_LONG", "ENTER_SHORT"}:
            self.last_processed_bar = bar_start
            self._save_state()
            return
        ema9 = float(latest["ema9"])
        if not math.isfinite(ema9):
            self.last_processed_bar = bar_start
            self._save_state()
            return
        self._open_position(
            action=decision.action,
            bar_start=bar_start,
            bar_close=bar_close.to_pydatetime(),
            quote_at=now,
            entry_index=float(latest["close"]),
            entry_ema9=ema9,
        )
        self.last_processed_bar = bar_start
        self._save_state()


def _live_preflight(execution: FyersExecutionClient, runner: EMA9ForwardRunner) -> None:
    if not execution.ensure_logged_in():
        raise RuntimeError("Fyers login failed; live execution is blocked.")
    try:
        master_rows = runner.api.option_master(runner.config.exchange)
    except (FyersAPIError, ValueError) as exc:
        raise RuntimeError("Fyers option master could not be loaded; live execution is blocked.") from exc
    if not master_rows:
        raise RuntimeError("Fyers contract master could not be loaded; live execution is blocked.")
    orders = execution.list_open_orders()
    positions = execution.list_open_positions()
    if orders.is_indeterminate or positions.is_indeterminate or orders.items:
        raise RuntimeError("Fyers account state is indeterminate or has unresolved orders.")
    broker_positions = list(positions.items)
    if runner.position is None:
        if broker_positions:
            raise RuntimeError("Fyers account is not flat; resolve positions before live startup.")
        return
    expected = runner.position
    if len(broker_positions) != 1:
        raise RuntimeError("Saved live position does not match the broker account; reconcile manually.")
    actual = broker_positions[0]
    if actual.symbol != expected.contract.symbol or actual.quantity != expected.quantity:
        raise RuntimeError("Saved live position does not match the broker account; reconcile manually.")


def _validate_live_resume(runner: EMA9ForwardRunner, *, now: datetime) -> None:
    position = runner.position
    if position is None:
        return
    if runner.trading_day != now.astimezone(IST).date() or now.astimezone(IST).time() >= SQUARE_OFF:
        raise RuntimeError(
            "A saved live position is outside its trading session. Reconcile and close it "
            "manually before restarting the EMA9 runner."
        )


def run_forward_runner(config: InstrumentConfig) -> None:
    load_dotenv(ENV_FILE, override=False)
    environment = dict(os.environ)
    notifier = _create_telegram_notifier(environment)
    mode = "PAPER"
    try:
        mode = execution_mode_from_environment(environment)
        _run_forward_runner(config, environment, notifier, mode=mode)
    except Exception as exc:
        _notify_runner_error(
            notifier,
            "Unrecoverable runner error",
            exc,
            instrument=config.name,
            mode=mode,
        )
        raise
    finally:
        notifier.close(timeout=10.0)


def _run_forward_runner(
    config: InstrumentConfig,
    environment: dict[str, str],
    notifier: TelegramNotifier,
    *,
    mode: str,
) -> None:
    poll_seconds = int(environment.get("EMA9_POLL_SECONDS", "5"))
    if poll_seconds < 5:
        raise ValueError("EMA9_POLL_SECONDS must be at least 5 to respect Fyers API pacing.")
    state_directory = environment.get("EMA9_STATE_DIR", "").strip()
    trades_directory = environment.get("EMA9_TRADES_DIR", "").strip()
    state_root = (
        Path(state_directory).expanduser()
        if state_directory
        else ROOT_DIR / "Forward Test Outputs"
    )
    trades_root = (
        Path(trades_directory).expanduser()
        if trades_directory
        else ROOT_DIR / "Forward Test Outputs"
    )
    if not state_root.is_absolute():
        state_root = ROOT_DIR / state_root
    if not trades_root.is_absolute():
        trades_root = ROOT_DIR / trades_root
    state_path = state_root / f"{config.name.lower()}_ema9_state.json"
    trades_path = trades_root / f"{config.name.lower()}_ema9_trades.csv"
    token_file_value = environment.get("FYERS_TOKEN_FILE", "").strip()
    token_file = (
        Path(token_file_value).expanduser()
        if token_file_value
        else ROOT_DIR / "Dependencies" / ".fyers_access_token"
    )
    if not token_file.is_absolute():
        token_file = ROOT_DIR / token_file
    app_id = environment.get("FYERS_APP_ID", "").strip()
    access_token = load_persisted_access_token(
        token_file, environment.get("FYERS_ACCESS_TOKEN", "")
    )
    api = FyersAPI(app_id, access_token or "telegram-auth-pending")
    signal_logic = _load_signal_logic()
    dashboard_state = FyersDashboardState()
    dashboard_enabled = environment.get("DASHBOARD_ENABLED", "true").strip().lower() == "true"
    dashboard_state.update_service(
        status="STARTING",
        mode=mode,
        strategy="EMA9",
        instrument=config.name,
        underlying=config.index_symbol,
    )

    def run_dashboard_backtest_for_range(start: date, end: date) -> dict[str, Any]:
        return run_dashboard_backtest(
            api,
            signal_logic,
            signal_logic.EMATrendConfig(),
            instrument=config.name,
            start=start,
            end=end,
        )

    auth_service = FyersTelegramAuthService(
        bot_token=environment.get("TELEGRAM_BOT_TOKEN", ""),
        authorized_chat_id=environment.get("FYERS_AUTH_TELEGRAM_CHAT_ID", ""),
        client_id=app_id,
        secret_key=environment.get("FYERS_SECRET_KEY", ""),
        redirect_uri=environment.get("FYERS_REDIRECT_URI", ""),
        token_file=token_file,
        market_data_client=api,
        port=int(environment.get("PORT", "8787")),
        dashboard_state=dashboard_state,
        dashboard_enabled=dashboard_enabled,
        dashboard_backtest_runner=run_dashboard_backtest_for_range,
    )
    execution = FyersExecutionClient(api)
    runner = EMA9ForwardRunner(
        api=api,
        execution=execution,
        config=config,
        mode=mode,
        state_path=state_path,
        trades_path=trades_path,
        notifier=notifier,
        dashboard_state=dashboard_state,
        signal_logic=signal_logic,
        strategy_config=signal_logic.EMATrendConfig(),
    )
    auth_service.start()
    try:
        auth_service.authenticate_startup(has_access_token=bool(access_token))
        if mode == "LIVE":
            _live_preflight(execution, runner)
            _validate_live_resume(runner, now=datetime.now(IST))
        dashboard_state.update_service(
            status="RUNNING",
            mode=mode,
            strategy="EMA9",
            instrument=config.name,
            underlying=config.index_symbol,
            started_at=datetime.now(IST).isoformat(),
        )
        notifier.notify(
            "✅ Strategy started successfully · EMA\n"
            f"Mode: {mode}\n"
            f"Underlying: {config.index_symbol}\n"
            "Lots: 1\n"
            f"Daily option P&L loss cap: ₹{runner.daily_loss_cap:g}\n"
            f"Contracts: nearest-upcoming-expiry 2-step ITM; BUY CE for LONG / BUY PE for SHORT\n"
            "Timeframe: completed five-minute candles; EMA9 exits use one-minute candles\n"
            f"Time: {datetime.now(IST):%Y-%m-%d %H:%M:%S} IST"
        )
        next_health_at = datetime.now(IST) + HEALTH_ALERT_INTERVAL
        market_data_status = "WAITING FOR FIRST FETCH"
        last_market_data_at: datetime | None = None
        LOGGER.info(
            "Starting %s EMA9 forward test in %s mode (2-step ITM, one lot, ₹%.0f daily cap).",
            config.name,
            mode,
            runner.daily_loss_cap,
        )
        warmup_day: date | None = None
        warmup_bars = pd.DataFrame()
        while True:
            now = datetime.now(IST)
            if not _is_market_data_session(now):
                sleep_time.sleep(poll_seconds)
                continue
            if now >= next_health_at:
                runner._notify(
                    _format_health_alert(
                        runner,
                        market_data_status=market_data_status,
                        last_market_data_at=last_market_data_at,
                        at=now,
                    )
                )
                next_health_at = now + HEALTH_ALERT_INTERVAL
            auth_service.wait_for_access_token()
            try:
                if warmup_day != now.date():
                    warmup_bars = api.history(
                        config.index_symbol,
                        resolution="1",
                        start=now.date() - timedelta(days=14),
                        end=now,
                    )
                    warmup_day = now.date()
                current_bars = api.history(
                    config.index_symbol,
                    resolution="1",
                    start=now.date(),
                    end=now,
                )
                market_data_status = "OK"
                last_market_data_at = now
                minute_bars = pd.concat([warmup_bars, current_bars], ignore_index=True)
                runner.process_market_data(minute_bars, now=now)
            except FyersAPIError as exc:
                market_data_status = "ERROR · market-data request failed"
                dashboard_state.update_market(status="ERROR")
                LOGGER.exception("%s EMA9 market-data request failed; retrying.", config.name)
                _notify_runner_error(
                    notifier,
                    "Market-data request failed; retrying",
                    exc,
                    instrument=config.name,
                    mode=mode,
                )
                try:
                    runner.check_open_position_risk(now=now)
                    runner.send_end_of_day_report(now=now)
                except FyersAPIError:
                    LOGGER.exception(
                        "%s EMA9 position risk could not be marked from a fresh option quote.",
                        config.name,
                    )
            except Exception as exc:
                market_data_status = "ERROR · processing cycle failed"
                dashboard_state.update_market(status="ERROR")
                LOGGER.exception("%s EMA9 processing cycle failed.", config.name)
                _notify_runner_error(
                    notifier,
                    "Processing cycle failed",
                    exc,
                    instrument=config.name,
                    mode=mode,
                )
                if mode == "LIVE":
                    raise
            sleep_delta = poll_seconds - (datetime.now(IST) - now).total_seconds()
            sleep_time.sleep(max(0.0, sleep_delta))
    finally:
        auth_service.stop()
