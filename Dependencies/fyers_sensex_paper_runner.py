"""Paper-only live-data runner for the SENSEX EMA9-touch strategy."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time as sleep_time
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Dependencies.fyers_api import FyersAPI, FyersAPIError  # noqa: E402
from Dependencies.fyers_ema_replay import ENV_FILE, IST, _load_signal_logic  # noqa: E402
from Dependencies.fyers_market_data import completed_five_minute_bars  # noqa: E402
from Dependencies.fyers_options import FyersOptionContract  # noqa: E402
from Dependencies.fyers_telegram_auth import (  # noqa: E402
    FyersTelegramAuthService,
    load_persisted_access_token,
)
from Dependencies.telegram_notifier import TelegramNotifier  # noqa: E402

INDEX_SYMBOL = "BSE:SENSEX-INDEX"
ENTRY_START = time(9, 25)
MARKET_OPEN = time(9, 15)
SQUARE_OFF = time(15, 15)
LOGGER = logging.getLogger("fyers_sensex_paper")
TRADE_COLUMNS = [
    "trade_id",
    "signal_bar_start",
    "entry_time",
    "exit_time",
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
    "index_points",
    "entry_option_price",
    "exit_option_price",
    "option_pnl",
]
STATE_VERSION = 3


@dataclass(slots=True)
class PaperPosition:
    signal: str
    signal_bar_start: str
    entry_time: str
    entry_index: float
    trailing_stop: float
    contract: FyersOptionContract
    quantity: int
    entry_option_price: float

    def __post_init__(self) -> None:
        if self.signal not in {"ENTER_LONG", "ENTER_SHORT"}:
            raise ValueError("Paper position signal must be ENTER_LONG or ENTER_SHORT.")
        if self.contract.option_type != ("CE" if self.signal == "ENTER_LONG" else "PE"):
            raise ValueError("Paper option type does not match the index signal direction.")
        if self.quantity != self.contract.lot_size:
            raise ValueError("SENSEX paper runner supports exactly one contract lot.")
        if not all(
            math.isfinite(value) and value > 0
            for value in (self.entry_index, self.trailing_stop, self.entry_option_price)
        ):
            raise ValueError("Paper position prices must be finite and positive.")


class SensexPaperEngine:
    """Paper ledger and strategy state; this class cannot submit broker orders."""

    def __init__(
        self,
        *,
        daily_loss_cap: float,
        trades_path: Path,
        state_path: Path,
        ema9_buffer_points: float = 0.0,
    ) -> None:
        if not math.isfinite(daily_loss_cap) or daily_loss_cap <= 0:
            raise ValueError("daily_loss_cap must be positive and finite.")
        if not math.isfinite(ema9_buffer_points) or ema9_buffer_points < 0:
            raise ValueError("ema9_buffer_points must be finite and non-negative.")
        self.daily_loss_cap = daily_loss_cap
        self.ema9_buffer_points = ema9_buffer_points
        self.trades_path = trades_path
        self.state_path = state_path
        self.position: PaperPosition | None = None
        self.daily_realized_points = 0.0
        self.daily_loss_halted = False
        self.trading_day: date | None = None
        self.last_processed_bar: pd.Timestamp | None = None
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("version") != STATE_VERSION:
                raise RuntimeError(
                    "SENSEX paper state is from an incompatible runner version. Archive the state and "
                    "ledger only after checking whether a paper position is still open."
                )
            saved_cap = float(state["daily_loss_cap"])
            if saved_cap != self.daily_loss_cap:
                raise RuntimeError(
                    "The SENSEX paper cap differs from the saved state. Resume with the original cap "
                    "or archive the state and trade ledger before starting a new paper run."
                )
            saved_buffer = float(state["ema9_buffer_points"])
            if saved_buffer != self.ema9_buffer_points:
                raise RuntimeError(
                    "The SENSEX EMA9 buffer differs from the saved state. Resume with the original buffer "
                    "or archive the state and trade ledger before starting a new paper run."
                )
            self.daily_realized_points = float(state["daily_realized_points"])
            self.daily_loss_halted = bool(state["daily_loss_halted"])
            self.trading_day = (
                date.fromisoformat(state["trading_day"]) if state.get("trading_day") else None
            )
            last_bar = state.get("last_processed_bar")
            self.last_processed_bar = pd.Timestamp(last_bar) if last_bar else None
            raw_position = state.get("position")
            if raw_position:
                raw_contract = raw_position.pop("contract")
                contract = FyersOptionContract(
                    symbol=raw_contract["symbol"],
                    expiry=date.fromisoformat(raw_contract["expiry"]),
                    option_type=raw_contract["option_type"],
                    strike=float(raw_contract["strike"]),
                    lot_size=int(raw_contract["lot_size"]),
                )
                self.position = PaperPosition(contract=contract, **raw_position)
        except RuntimeError:
            raise
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to read SENSEX paper state {self.state_path}.") from exc
        if not math.isfinite(self.daily_realized_points):
            raise RuntimeError(f"SENSEX paper state has invalid realized points: {self.state_path}.")

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": STATE_VERSION,
            "daily_loss_cap": self.daily_loss_cap,
            "ema9_buffer_points": self.ema9_buffer_points,
            "daily_realized_points": self.daily_realized_points,
            "daily_loss_halted": self.daily_loss_halted,
            "trading_day": self.trading_day.isoformat() if self.trading_day else None,
            "last_processed_bar": self.last_processed_bar.isoformat() if self.last_processed_bar else None,
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
                if self.position
                else None
            ),
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.state_path.parent,
                prefix=f".{self.state_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(state, temporary, sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.state_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def initialize(self, latest_bar_time: pd.Timestamp, today: date) -> bool:
        has_open_position = self.position is not None
        if self.position is not None:
            position_day = pd.Timestamp(self.position.entry_time).date()
            if position_day != today:
                raise RuntimeError(
                    "A SENSEX paper position from a prior day is still open in saved state. "
                    "Inspect the paper ledger and state file before continuing."
                )
            if self.last_processed_bar is None or self.trading_day != today:
                raise RuntimeError(
                    "Saved SENSEX position is missing its same-day processing cursor. "
                    "Inspect the state and ledger before continuing."
                )
        if self.trading_day != today or self.last_processed_bar is None:
            self.trading_day = today
            self.daily_realized_points = 0.0
            self.daily_loss_halted = False
        if not has_open_position:
            self.last_processed_bar = latest_bar_time
        self.save_state()
        return has_open_position

    def _append_trade(self, trade: dict[str, Any]) -> None:
        self.trades_path.parent.mkdir(parents=True, exist_ok=True)
        if self.trades_path.exists():
            existing_columns = pd.read_csv(self.trades_path, nrows=0).columns.tolist()
            if existing_columns != TRADE_COLUMNS:
                raise RuntimeError(
                    f"SENSEX paper ledger columns do not match this runner: {self.trades_path}."
                )
            existing_ids = pd.read_csv(self.trades_path, usecols=["trade_id"])["trade_id"]
            if trade["trade_id"] in set(existing_ids):
                return
        pd.DataFrame([trade], columns=TRADE_COLUMNS).to_csv(
            self.trades_path,
            mode="a",
            header=not self.trades_path.exists(),
            index=False,
        )

    def close_position(
        self,
        at: pd.Timestamp,
        index_price: float,
        option_price: float,
        reason: str,
    ) -> dict[str, Any]:
        if self.position is None:
            raise RuntimeError("Cannot close a SENSEX paper position when none is open.")
        if not math.isfinite(index_price) or index_price <= 0:
            raise ValueError("SENSEX paper exit index price must be finite and positive.")
        if not math.isfinite(option_price) or option_price <= 0:
            raise ValueError("SENSEX paper exit option price must be finite and positive.")
        closed_position = self.position
        direction = 1.0 if closed_position.signal == "ENTER_LONG" else -1.0
        option_pnl = (option_price - closed_position.entry_option_price) * closed_position.quantity
        trade = {
            "trade_id": hashlib.sha256(
                f"{closed_position.entry_time}|{closed_position.contract.symbol}".encode()
            ).hexdigest(),
            "signal_bar_start": closed_position.signal_bar_start,
            "entry_time": closed_position.entry_time,
            "exit_time": at.isoformat(),
            "signal": closed_position.signal,
            "exit_reason": reason,
            "option_symbol": closed_position.contract.symbol,
            "option_expiry": closed_position.contract.expiry.isoformat(),
            "option_type": closed_position.contract.option_type,
            "strike": closed_position.contract.strike,
            "lot_size": closed_position.contract.lot_size,
            "quantity": closed_position.quantity,
            "entry_index": closed_position.entry_index,
            "exit_index": index_price,
            "index_points": direction * (index_price - closed_position.entry_index),
            "entry_option_price": closed_position.entry_option_price,
            "exit_option_price": option_price,
            "option_pnl": option_pnl,
        }
        self.daily_realized_points += trade["index_points"]
        self._append_trade(trade)
        self.position = None
        if self.daily_realized_points <= -self.daily_loss_cap:
            self.daily_loss_halted = True
        self.save_state()
        LOGGER.info(
            "PAPER EXIT %s %s; option pnl=%+.2f INR; index=%+.2f points; daily index=%+.2f; reason=%s",
            trade["signal"],
            closed_position.contract.symbol,
            option_pnl,
            trade["index_points"],
            self.daily_realized_points,
            reason,
        )
        return trade

    def process_completed_bars(
        self,
        enriched_bars: pd.DataFrame,
        minute_bars: pd.DataFrame,
        *,
        start: date,
        end: date,
        now: datetime,
        signal_logic: Any,
        strategy_config: Any,
        api: FyersAPI,
        notifier: TelegramNotifier,
    ) -> None:
        bar_times = pd.to_datetime(enriched_bars["timestamp"], errors="coerce")
        bar_times = (
            bar_times.dt.tz_localize(IST)
            if bar_times.dt.tz is None
            else bar_times.dt.tz_convert(IST)
        )
        minutes = minute_bars.copy()
        minutes["timestamp"] = pd.to_datetime(minutes["timestamp"], errors="coerce")
        minutes["timestamp"] = (
            minutes["timestamp"].dt.tz_localize(IST)
            if minutes["timestamp"].dt.tz is None
            else minutes["timestamp"].dt.tz_convert(IST)
        )
        completed_before = pd.Timestamp(now).floor("min")
        minutes = minutes.dropna(subset=["timestamp"])
        minutes = minutes[minutes["timestamp"] + pd.Timedelta(minutes=1) <= completed_before]
        minutes = minutes[minutes["timestamp"].dt.date.between(start, end)].copy()
        minutes["bar_start"] = minutes["timestamp"].dt.floor("5min")
        minutes_by_bar = {key: group for key, group in minutes.groupby("bar_start", sort=True)}  # noqa: C416
        engine = signal_logic.EMATrendSignalEngine(strategy_config)

        for bar_index in range(len(enriched_bars)):
            bar = enriched_bars.iloc[bar_index]
            bar_start = bar_times.iloc[bar_index]
            trading_day = bar_start.date()
            if not start <= trading_day <= end:
                continue
            if self.last_processed_bar is not None and bar_start <= self.last_processed_bar:
                continue
            if self.trading_day != trading_day:
                if self.position is not None:
                    raise RuntimeError("Paper position crossed into a new SENSEX trading day.")
                self.trading_day = trading_day
                self.daily_realized_points = 0.0
                self.daily_loss_halted = False

            bar_close_time = bar_start + pd.Timedelta(minutes=5)
            position_open_at_bar_start = self.position is not None
            if self.position is not None:
                direction = 1.0 if self.position.signal == "ENTER_LONG" else -1.0
                minute_group = minutes_by_bar.get(bar_start)
                if minute_group is not None and bar_start > pd.Timestamp(self.position.signal_bar_start):
                    prior_ema9 = float(
                        enriched_bars.iloc[bar_index - 1]["ema9"]
                    ) if bar_index > 0 else math.nan
                    direction = 1.0 if self.position.signal == "ENTER_LONG" else -1.0
                    prior_stop = prior_ema9 - direction * self.ema9_buffer_points
                    for minute in minute_group.itertuples(index=False):
                        if self.position is None:
                            break
                        minute_close_time = minute.timestamp + pd.Timedelta(minutes=1)
                        if minute.timestamp < pd.Timestamp(self.position.entry_time):
                            continue

                        if minute_close_time == bar_close_time:
                            direction = 1.0 if self.position.signal == "ENTER_LONG" else -1.0
                            close_points = direction * (
                                float(bar["close"]) - self.position.entry_index
                            )
                            exit_reason = ""
                            exit_index = float(bar["close"])
                            if self.daily_realized_points + close_points <= -self.daily_loss_cap:
                                exit_reason = "DAILY_MAX_LOSS"
                                self.daily_loss_halted = True
                            else:
                                trailing_stop = self.position.trailing_stop
                                trail_hit = (
                                    float(bar["close"]) <= trailing_stop
                                    if direction > 0
                                    else float(bar["close"]) >= trailing_stop
                                )
                                if trail_hit:
                                    exit_reason = "TRAILING_STOP"
                                    if self.daily_realized_points + close_points <= -self.daily_loss_cap:
                                        self.daily_loss_halted = True
                                elif bar_close_time.time() >= SQUARE_OFF:
                                    exit_reason = "SQUARE_OFF"

                            if exit_reason:
                                closing_position = self.position
                                option_price = api.quote_ltp(closing_position.contract.symbol)
                                trade = self.close_position(
                                    bar_close_time,
                                    exit_index,
                                    option_price,
                                    exit_reason,
                                )
                                notifier.notify(_format_exit_message(closing_position, trade))
                                if exit_reason == "DAILY_MAX_LOSS":
                                    break
                            else:
                                current_ema9 = float(bar["ema9"])
                                if math.isfinite(current_ema9):
                                    buffered_ema9 = current_ema9 - direction * self.ema9_buffer_points
                                    self.position.trailing_stop = (
                                        max(self.position.trailing_stop, buffered_ema9)
                                        if direction > 0
                                        else min(self.position.trailing_stop, buffered_ema9)
                                    )

                        if self.position is None or not math.isfinite(prior_stop):
                            continue
                        if self.position.signal == "ENTER_LONG" and float(minute.low) <= prior_stop:
                            exit_index = (
                                float(minute.open)
                                if float(minute.open) <= prior_stop
                                else prior_stop
                            )
                        elif (
                            self.position.signal == "ENTER_SHORT"
                            and float(minute.high) >= prior_stop
                        ):
                            exit_index = (
                                float(minute.open)
                                if float(minute.open) >= prior_stop
                                else prior_stop
                            )
                        else:
                            continue
                        closing_position = self.position
                        option_price = api.quote_ltp(closing_position.contract.symbol)
                        trade = self.close_position(
                            minute_close_time,
                            exit_index,
                            option_price,
                            "EMA9_TOUCH",
                        )
                        notifier.notify(_format_exit_message(closing_position, trade))

            if (
                self.position is None
                and not position_open_at_bar_start
                and not self.daily_loss_halted
                and self.daily_realized_points > -self.daily_loss_cap
                and ENTRY_START <= bar_close_time.time() < SQUARE_OFF
            ):
                decision = engine.evaluate_candle(enriched_bars.iloc[: bar_index + 1])
                if decision.action in {"ENTER_LONG", "ENTER_SHORT"}:
                    entry_ema9 = float(bar["ema9"])
                    if math.isfinite(entry_ema9):
                        direction_sign = 1.0 if decision.action == "ENTER_LONG" else -1.0
                        contract = api.resolve_nearest_option(
                            decision.action,
                            spot=float(bar["close"]),
                            today=trading_day,
                        )
                        option_price = api.quote_ltp(contract.symbol)
                        direction = "LONG" if decision.action == "ENTER_LONG" else "SHORT"
                        self.position = PaperPosition(
                            signal=decision.action,
                            signal_bar_start=bar_start.isoformat(),
                            entry_time=bar_close_time.isoformat(),
                            entry_index=float(bar["close"]),
                            trailing_stop=entry_ema9 - direction_sign * self.ema9_buffer_points,
                            contract=contract,
                            quantity=contract.lot_size,
                            entry_option_price=option_price,
                        )
                        self.last_processed_bar = bar_start
                        self.save_state()
                        notifier.notify(_format_entry_message(self.position))
                        LOGGER.info(
                            "PAPER ENTRY %s %s at option LTP %.2f; index=%.2f; %s",
                            direction,
                            contract.symbol,
                            option_price,
                            float(bar["close"]),
                            bar_close_time.isoformat(),
                        )
            self.last_processed_bar = bar_start
            self.save_state()

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=float, help="Daily loss cap in SENSEX points.")
    parser.add_argument(
        "--ema9-buffer-points",
        type=float,
        help="EMA9 stop cushion in SENSEX points (default: 3).",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        help="Live quote polling interval in seconds.",
    )
    parser.add_argument(
        "--trades-file",
        type=Path,
        help="Paper trade ledger CSV path.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        help="Resumable paper position state path.",
    )
    return parser


def _format_entry_message(position: PaperPosition) -> str:
    contract = position.contract
    direction = "LONG" if position.signal == "ENTER_LONG" else "SHORT"
    symbol = f"SENSEX-{contract.expiry:%b%Y}-{contract.strike:g}-{contract.option_type}"
    return (
        "🚀 ENTRY · EMA\n"
        f"Direction: {direction}\n"
        f"Instrument: BUY {symbol}\n"
        f"  Entry: ₹{position.entry_option_price:.2f}\n"
        f"Qty: {position.quantity} (1 lot(s) x {contract.lot_size})\n"
        f"Index entry: {position.entry_index:.2f}\n"
        f"[PAPER] {pd.Timestamp(position.entry_time).tz_convert(IST):%Y-%m-%d %H:%M:%S} IST"
    )


def _format_exit_message(position: PaperPosition, trade: dict[str, Any]) -> str:
    contract = position.contract
    direction = "LONG" if position.signal == "ENTER_LONG" else "SHORT"
    symbol = f"SENSEX-{contract.expiry:%b%Y}-{contract.strike:g}-{contract.option_type}"
    display_reason = "TIME_CUTOFF" if trade["exit_reason"] == "SQUARE_OFF" else trade["exit_reason"]
    option_pnl = trade["option_pnl"]
    pnl_text = f"+₹{option_pnl:.2f}" if option_pnl >= 0 else f"-₹{abs(option_pnl):.2f}"
    return (
        f"🏁 EXIT · EMA ({display_reason})\n"
        f"Direction: {direction}\n"
        f"Instrument: SELL {symbol}\n"
        f"  Entry: ₹{trade['entry_option_price']:.2f} → Exit: ₹{trade['exit_option_price']:.2f}\n"
        f"Qty: {position.quantity} (lot size {contract.lot_size})\n"
        f"⚪ P&L: {pnl_text}\n"
        f"Index points: {trade['index_points']:+.2f}\n"
        f"[PAPER] {pd.Timestamp(trade['exit_time']).tz_convert(IST):%Y-%m-%d %H:%M:%S} IST"
    )


def _require_paper_mode() -> None:
    mode = os.getenv("LIVE_TRADING_ENABLED", "").strip().lower()
    if mode != "false":
        raise RuntimeError(
            "SENSEX option runner is paper-only and requires LIVE_TRADING_ENABLED=false explicitly."
        )


def _authenticate_fyers_through_telegram() -> tuple[FyersAPI, FyersTelegramAuthService]:
    app_id = os.getenv("FYERS_APP_ID", "").strip()
    if not app_id:
        raise RuntimeError("Set FYERS_APP_ID for Fyers authentication.")
    token_file_value = os.getenv("FYERS_TOKEN_FILE", "").strip()
    token_file = (
        Path(token_file_value).expanduser()
        if token_file_value
        else ROOT_DIR / "Dependencies" / ".fyers_access_token"
    )
    if not token_file.is_absolute():
        token_file = ROOT_DIR / token_file
    access_token = load_persisted_access_token(token_file, os.getenv("FYERS_ACCESS_TOKEN", ""))
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
    )
    try:
        auth_service.start()
        auth_service.authenticate_startup(has_access_token=bool(access_token))
    except BaseException:
        auth_service.stop()
        raise
    return api, auth_service


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_dotenv(ENV_FILE, override=False)
    _require_paper_mode()
    args.cap = (
        args.cap
        if args.cap is not None
        else float(os.getenv("SENSEX_PAPER_DAILY_LOSS_CAP", "300"))
    )
    args.ema9_buffer_points = (
        args.ema9_buffer_points
        if args.ema9_buffer_points is not None
        else float(os.getenv("SENSEX_PAPER_EMA9_BUFFER_POINTS", "3"))
    )
    args.poll_seconds = (
        args.poll_seconds
        if args.poll_seconds is not None
        else int(os.getenv("SENSEX_PAPER_POLL_SECONDS", "30"))
    )
    if not math.isfinite(args.cap) or args.cap <= 0:
        raise SystemExit("--cap must be positive and finite.")
    if not math.isfinite(args.ema9_buffer_points) or args.ema9_buffer_points < 0:
        raise SystemExit("--ema9-buffer-points must be finite and non-negative.")
    if args.poll_seconds <= 0 or args.poll_seconds > 120:
        raise SystemExit("--poll-seconds must be between 1 and 120.")
    args.trades_file = args.trades_file or Path(
        os.getenv("SENSEX_PAPER_TRADES_FILE", "").strip()
        or str(ROOT_DIR / "Backtest Outputs" / "fyers_sensex_paper_trades.csv")
    ).expanduser()
    args.state_file = args.state_file or Path(
        os.getenv("SENSEX_PAPER_STATE_FILE", "").strip()
        or str(ROOT_DIR / "Backtest Outputs" / ".fyers_sensex_paper_state.json")
    ).expanduser()
    if os.getenv("TELEGRAM_ENABLED", "").strip().lower() != "true":
        raise RuntimeError("Set TELEGRAM_ENABLED=true to receive required SENSEX paper trade alerts.")
    if not os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or not os.getenv("TELEGRAM_CHAT_ID", "").strip():
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID for SENSEX paper alerts.")
    notifier = TelegramNotifier.from_environment()
    atexit.register(notifier.close, timeout=10.0)
    api, auth_service = _authenticate_fyers_through_telegram()
    atexit.register(auth_service.stop)
    logic = _load_signal_logic()
    strategy_config = logic.EMATrendConfig(exit_ema_period=9)
    engine = SensexPaperEngine(
        daily_loss_cap=args.cap,
        ema9_buffer_points=args.ema9_buffer_points,
        trades_path=args.trades_file,
        state_path=args.state_file,
    )
    LOGGER.info(
        "SENSEX EMA9-touch PAPER options runner started; cap=%s index points; EMA9 buffer=%s points; "
        "poll=%ss; no orders will be submitted.",
        args.cap,
        args.ema9_buffer_points,
        args.poll_seconds,
    )
    notifier.notify(
        "✅ Strategy started successfully · EMA\n"
        "Mode: PAPER\n"
        f"Underlying: {INDEX_SYMBOL}\n"
        f"Daily loss cap: {args.cap:g} SENSEX index points\n"
        f"EMA9 stop buffer: {args.ema9_buffer_points:g} SENSEX points\n"
        "Contracts: nearest-expiry ATM; BUY CE for LONG / BUY PE for SHORT\n"
        "Timeframe: completed five-minute candles"
    )
    initialized = False

    def finish_session(index_price: float | None = None) -> int:
        if engine.position is not None:
            closing_position = engine.position
            if index_price is None:
                index_price = api.quote_ltp(INDEX_SYMBOL)
            option_price = api.quote_ltp(closing_position.contract.symbol)
            trade = engine.close_position(
                pd.Timestamp.now(tz=IST),
                index_price,
                option_price,
                "SQUARE_OFF",
            )
            notifier.notify(_format_exit_message(closing_position, trade))
        LOGGER.info("SENSEX paper session ended at square-off.")
        auth_service.stop()
        atexit.unregister(auth_service.stop)
        notifier.close()
        atexit.unregister(notifier.close)
        return 0

    while True:
        now = datetime.now(IST)
        market_time = now.timetz().replace(tzinfo=None)
        if market_time < MARKET_OPEN:
            sleep_seconds = min(
                args.poll_seconds,
                max(1.0, (datetime.combine(now.date(), MARKET_OPEN, tzinfo=IST) - now).total_seconds()),
            )
            sleep_time.sleep(sleep_seconds)
            continue
        try:
            minute_history = api.history(
                INDEX_SYMBOL,
                resolution="1",
                start=now.date() - timedelta(days=7),
                end=now.date(),
            )
            if minute_history.empty:
                if market_time >= SQUARE_OFF:
                    return finish_session()
                LOGGER.info("No SENSEX minute history yet; retrying.")
                sleep_time.sleep(args.poll_seconds)
                continue
            five_minute_bars = completed_five_minute_bars(minute_history, now)
            if five_minute_bars.empty:
                if market_time >= SQUARE_OFF:
                    return finish_session()
                sleep_time.sleep(args.poll_seconds)
                continue
            enriched = logic.build_ema_trend_with_indicators(five_minute_bars, strategy_config)
            latest_bar_time = pd.Timestamp(enriched.iloc[-1]["timestamp"])
            if latest_bar_time.tzinfo is None:
                latest_bar_time = latest_bar_time.tz_localize(IST)
            else:
                latest_bar_time = latest_bar_time.tz_convert(IST)
            process_bars = initialized
            if not initialized:
                restored_position = engine.initialize(latest_bar_time, now.date())
                initialized = True
                process_bars = restored_position
            if process_bars:
                engine.process_completed_bars(
                    enriched,
                    minute_history,
                    start=now.date(),
                    end=now.date(),
                    now=now,
                    signal_logic=logic,
                    strategy_config=strategy_config,
                    api=api,
                    notifier=notifier,
                )
            index_quote = api.quote_ltp(INDEX_SYMBOL)
            option_open_pnl = 0.0
            if engine.position is not None:
                option_quote = api.quote_ltp(engine.position.contract.symbol)
                option_open_pnl = (
                    option_quote - engine.position.entry_option_price
                ) * engine.position.quantity
            engine.save_state()
            open_points = 0.0
            if engine.position is not None:
                direction = 1.0 if engine.position.signal == "ENTER_LONG" else -1.0
                open_points = direction * (index_quote - engine.position.entry_index)
            LOGGER.info(
                (
                    "SENSEX=%.2f; EMA9=%.2f; realized=%+.2f index pts; "
                    "open=%+.2f index pts; option open P&L=%+.2f INR; cap=%s; halted=%s"
                ),
                index_quote,
                float(enriched.iloc[-1]["ema9"]),
                engine.daily_realized_points,
                open_points,
                option_open_pnl,
                args.cap,
                engine.daily_loss_halted,
            )
            if market_time >= SQUARE_OFF:
                return finish_session(index_quote)
        except FyersAPIError as exc:
            LOGGER.error("Fyers market-data request failed; paper runner will retry: %s", exc)
        sleep_time.sleep(args.poll_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
