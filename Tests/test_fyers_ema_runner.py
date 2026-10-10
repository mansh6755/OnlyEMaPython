from datetime import date, datetime
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from Dependencies import fyers_ema_runner as ema_runner
from Dependencies.broker_contract import normalize_order_result
from Dependencies.fyers_ema_runner import (
    ActivePosition,
    ClosedPosition,
    _close_position,
    _dashboard_candles,
    _dashboard_position,
    _entry_bar_is_fresh,
    _format_day_started_alert,
    _format_entry_alert,
    _format_eod_alert,
    _format_error_alert,
    _format_exit_alert,
    _format_health_alert,
    _format_pnl,
    _format_strategy_started_alert,
    completed_five_minute_bars,
    daily_loss_limit_reached,
)
from Dependencies.fyers_options import FyersOptionContract

IST = ZoneInfo("Asia/Kolkata")


def test_completed_five_minute_bars_aggregates_only_closed_full_bars():
    timestamps = pd.date_range("2026-10-09 09:15", periods=7, freq="min", tz=IST)
    minute_bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100, 101, 102, 103, 104, 105, 106],
            "high": [101, 102, 103, 104, 105, 106, 107],
            "low": [99, 100, 101, 102, 103, 104, 105],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5],
            "volume": [10, 11, 12, 13, 14, 15, 16],
        }
    )

    result = completed_five_minute_bars(
        minute_bars,
        datetime(2026, 10, 9, 9, 22, tzinfo=IST),
    )

    assert len(result) == 1
    assert result.iloc[0]["timestamp"] == pd.Timestamp("2026-10-09 09:15", tz=IST)
    assert result.iloc[0]["open"] == 100
    assert result.iloc[0]["high"] == 105
    assert result.iloc[0]["low"] == 99
    assert result.iloc[0]["close"] == 104.5
    assert result.iloc[0]["volume"] == 60


def test_daily_loss_limit_includes_unrealized_option_loss():
    assert daily_loss_limit_reached(
        realized_pnl=-1000,
        entry_price=100,
        mark_price=30,
        quantity=65,
        max_daily_loss=5500,
    )


def test_daily_loss_limit_allows_trades_above_the_configured_floor():
    assert not daily_loss_limit_reached(
        realized_pnl=-1000,
        entry_price=100,
        mark_price=50,
        quantity=65,
        max_daily_loss=5500,
    )


def test_entry_bar_freshness_is_bounded_to_two_minutes_after_close():
    timestamp = pd.Timestamp("2026-10-09 09:20", tz=IST)

    assert _entry_bar_is_fresh(timestamp, datetime(2026, 10, 9, 9, 26, tzinfo=IST))
    assert not _entry_bar_is_fresh(timestamp, datetime(2026, 10, 9, 9, 28, tzinfo=IST))


def test_live_square_off_submits_before_needing_a_quote():
    class FakeAPI:
        def quote_ltp(self, symbol):
            raise AssertionError("A confirmed broker fill price should avoid quote fallback.")

    class FakeExecution:
        def place_market_order(self, *args, **kwargs):
            return normalize_order_result(
                order_id="exit-1",
                requested_quantity=65,
                filled_quantity=65,
                broker_state="FILLED",
                average_fill_price=90,
            )

    position = ActivePosition(
        direction="LONG",
        contract=FyersOptionContract(
            "NSE:NIFTY26O1324000CE", date(2026, 10, 13), "CE", 24000, 65
        ),
        quantity=65,
        entry_underlying=24000,
        stop_underlying=23950,
        entry_option_price=100,
        placed_at=datetime(2026, 10, 9, 15, 14, 59, tzinfo=IST),
    )

    closed = _close_position(
        FakeAPI(),
        FakeExecution(),
        position,
        live_enabled=True,
        product_type="INTRADAY",
        reason="SQUARE_OFF",
    )

    assert closed.pnl == -650
    assert closed.exit_option_price == 90
    assert closed.placed_at.tzinfo == IST


def test_telegram_trade_alerts_match_requested_layout_and_show_placement_time():
    position = ActivePosition(
        direction="SHORT",
        contract=FyersOptionContract(
            "NSE:NIFTY26O1323350PE", date(2026, 10, 13), "PE", 23350, 65
        ),
        quantity=65,
        entry_underlying=23400,
        stop_underlying=23450,
        entry_option_price=759.0,
        placed_at=datetime(2026, 10, 10, 9, 19, 6, tzinfo=IST),
    )
    closed = ClosedPosition(
        exit_option_price=759.0,
        pnl=0.0,
        placed_at=datetime(2026, 10, 10, 9, 45, 0, tzinfo=IST),
    )

    entry = _format_entry_alert(position, lots=1, mode="PAPER")
    exit_alert = _format_exit_alert(position, closed, "SQUARE_OFF", "PAPER")
    eod = _format_eod_alert(0.0, 1, "PAPER", datetime(2026, 10, 10, 9, 45, 1, tzinfo=IST))

    assert "Direction: SHORT" in entry
    assert "Instrument: BUY NIFTY-Oct2026-23350-PE" in entry
    assert "Entry: ₹759.00" in entry
    assert "Qty: 65 (1 lot(s) x 65)" in entry
    assert "[PAPER] 2026-10-10 09:19:06 IST" in entry
    assert "EXIT · EMA (TIME_CUTOFF)" in exit_alert
    assert "Entry: ₹759.00 → Exit: ₹759.00" in exit_alert
    assert "Instrument: SELL NIFTY-Oct2026-23350-PE" in exit_alert
    assert "P&L: +₹0.00" in exit_alert
    assert "[PAPER] 2026-10-10 09:45:00 IST" in exit_alert
    assert "EMA [PAPER]: +₹0.00 (1 trade(s))" in eod
    assert "Total: +₹0.00 (1 trade(s))" in eod
    assert _format_pnl(-12.5) == "-₹12.50"


def test_error_alert_includes_traceback_and_redacts_access_token(monkeypatch):
    secret = "fyers-access-secret-123"
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", secret)
    try:
        raise RuntimeError(f"request failed with access_token={secret}")
    except RuntimeError as error:
        alert = _format_error_alert("Market-data failure", error, "LIVE")

    assert "🚨 ERROR · EMA" in alert
    assert "RuntimeError" in alert
    assert "Traceback (most recent call last)" in alert
    assert secret not in alert
    assert "<redacted>" in alert
    assert len(alert) < 4096


def test_startup_day_started_and_hourly_health_alert_contents():
    startup_at = datetime(2026, 10, 12, 9, 20, tzinfo=IST)
    fetch_at = datetime(2026, 10, 12, 9, 25, tzinfo=IST)
    heartbeat_at = datetime(2026, 10, 12, 10, 25, tzinfo=IST)

    startup = _format_strategy_started_alert("PAPER", "NSE:NIFTY50-INDEX", 1, startup_at)
    day_started = _format_day_started_alert("PAPER", startup_at.date(), 375, 75, fetch_at)
    health = _format_health_alert("PAPER", None, 125.5, "OK", fetch_at, heartbeat_at)

    assert "Strategy started successfully" in startup
    assert "Underlying: NSE:NIFTY50-INDEX" in startup
    assert "Lots: 1" in startup
    assert "Day started successfully" in day_started
    assert "fetched 375 one-minute bar(s)" in day_started
    assert "75 completed five-minute bar(s)" in day_started
    assert "HEALTH · EMA" in health
    assert "Status: RUNNING" in health
    assert "Market data: OK" in health
    assert "Position: FLAT" in health
    assert "Realized P&L today: +₹125.50" in health


def test_dashboard_projects_chart_candles_and_open_exposure():
    timestamp = pd.Timestamp("2026-10-12 09:25:00", tz=IST)
    frame = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "open": 24000.0,
                "high": 24020.0,
                "low": 23995.0,
                "close": 24010.0,
                "ema4": 24005.0,
                "ema11": 23998.0,
                "ema18": 23990.0,
            }
        ]
    )
    position = ActivePosition(
        direction="LONG",
        contract=FyersOptionContract(
            "NSE:NIFTY26O1324000CE", date(2026, 10, 13), "CE", 24000, 65
        ),
        quantity=65,
        entry_underlying=24010.0,
        stop_underlying=23998.0,
        entry_option_price=100.0,
        placed_at=datetime(2026, 10, 12, 9, 26, tzinfo=IST),
    )

    candle = _dashboard_candles(frame)[0]
    exposure = _dashboard_position(position, 112.0)

    assert candle["time"] == int(timestamp.timestamp())
    assert candle["close"] == 24010.0
    assert candle["ema_mid"] == 23998.0
    assert exposure["display_symbol"] == "NIFTY-Oct2026-24000-CE"
    assert exposure["open_pnl"] == 780.0
    assert _dashboard_position(None, None) is None


def test_dashboard_projects_ema_chart_and_open_position_data():
    timestamp = pd.Timestamp("2026-10-12 09:25:00", tz=IST)
    frame = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "open": 24000.0,
                "high": 24020.0,
                "low": 23995.0,
                "close": 24010.0,
                "ema4": 24005.0,
                "ema11": 23998.0,
                "ema18": 23990.0,
            }
        ]
    )
    candle = _dashboard_candles(frame)[0]
    position = ActivePosition(
        direction="LONG",
        contract=FyersOptionContract("NSE:NIFTY26O1324000CE", date(2026, 10, 13), "CE", 24000, 65),
        quantity=65,
        entry_underlying=24010.0,
        stop_underlying=23998.0,
        entry_option_price=100.0,
        placed_at=datetime(2026, 10, 12, 9, 26, tzinfo=IST),
    )
    dashboard_position = _dashboard_position(position, 112.0)

    assert candle["time"] == int(timestamp.timestamp())
    assert candle["close"] == 24010.0
    assert candle["ema_mid"] == 23998.0
    assert dashboard_position["display_symbol"] == "NIFTY-Oct2026-24000-CE"
    assert dashboard_position["open_pnl"] == 780.0
    assert _dashboard_position(None, None) is None


def test_runner_reports_fatal_error_to_telegram_then_reraises(monkeypatch):
    notifier = Mock()
    auth_service = Mock()
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    monkeypatch.setenv("FYERS_APP_ID", "test-app-id")
    monkeypatch.setenv("FYERS_SECRET_KEY", "test-app-secret")
    monkeypatch.setenv("FYERS_REDIRECT_URI", "https://service.example/callback/")
    monkeypatch.setenv("FYERS_AUTH_TELEGRAM_CHAT_ID", "12345")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token")
    monkeypatch.setattr(ema_runner, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ema_runner.TelegramNotifier, "from_environment", lambda: notifier)
    monkeypatch.setattr(ema_runner, "load_persisted_access_token", lambda *_args: "valid-test-token")
    monkeypatch.setattr(ema_runner, "FyersAPI", lambda *_args: object())
    monkeypatch.setattr(ema_runner, "FyersTelegramAuthService", lambda **_kwargs: auth_service)

    def fail_runner(*_args):
        raise RuntimeError("runner exploded")

    monkeypatch.setattr(ema_runner, "_run_ema_runner", fail_runner)

    with pytest.raises(RuntimeError, match="runner exploded"):
        ema_runner.run_ema_runner(None, None)

    alert = notifier.notify_error.call_args.args[1]
    assert "Unrecoverable runner error" in alert
    assert "runner exploded" in alert
    notifier.close.assert_called_once_with(timeout=10.0)
    auth_service.start.assert_called_once()
    auth_service.authenticate_startup.assert_called_once_with(has_access_token=True)
    auth_service.stop.assert_called_once()
