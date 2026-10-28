import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from Dependencies.fyers_api import FyersAPI, FyersAPIError
from Dependencies.broker_contract import OrderStatus
from Dependencies.fyers_dashboard_state import FyersDashboardState
from Dependencies.fyers_ema9_forward_launcher import selected_instrument
from Dependencies.fyers_ema9_forward_runner import (
    EMA9ForwardRunner,
    FyersQuoteError,
    ForwardPosition,
    InstrumentConfig,
    _create_telegram_notifier,
    _clear_overnight_caches,
    _format_health_alert,
    _is_market_data_session,
    _latest_trading_day,
    _quote_option_price,
    _validate_live_resume,
    execution_mode_from_environment,
    select_itm_contract,
)
from Dependencies.fyers_options import FyersOptionContract

IST = ZoneInfo("Asia/Kolkata")


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, "PAPER"),
        ({"EMA9_EXECUTION_MODE": "paper"}, "PAPER"),
        ({"EMA9_EXECUTION_MODE": "LIVE"}, "LIVE"),
    ],
)
def test_execution_mode_defaults_to_paper_and_accepts_explicit_live(environment, expected):
    assert execution_mode_from_environment(environment) == expected


def test_invalid_execution_mode_fails_closed():
    with pytest.raises(ValueError, match="PAPER or LIVE"):
        execution_mode_from_environment({"EMA9_EXECUTION_MODE": "AUTO"})


def test_telegram_notifier_uses_auth_chat_as_trade_alert_fallback(monkeypatch):
    created = {}

    class FakeNotifier:
        def __init__(self, bot_token, chat_id, *, enabled):
            created.update(bot_token=bot_token, chat_id=chat_id, enabled=enabled)

    monkeypatch.setattr(
        "Dependencies.fyers_ema9_forward_runner.TelegramNotifier", FakeNotifier
    )

    _create_telegram_notifier(
        {
            "TELEGRAM_BOT_TOKEN": "bot-token",
            "FYERS_AUTH_TELEGRAM_CHAT_ID": "private-chat-id",
        }
    )

    assert created == {
        "bot_token": "bot-token",
        "chat_id": "private-chat-id",
        "enabled": True,
    }


def test_telegram_notifier_rejects_missing_alert_destination():
    with pytest.raises(ValueError, match="Telegram alerts require"):
        _create_telegram_notifier({"TELEGRAM_BOT_TOKEN": "bot-token"})


def test_hourly_health_alert_reports_runner_and_market_status(tmp_path):
    runner = _forward_runner(tmp_path, _FakeApi(), _NoOrderExecution())
    runner.trading_day = date(2026, 10, 8)
    runner.daily_realized_option_pnl = 125.5
    runner.daily_trade_count = 2
    at = datetime(2026, 10, 8, 11, 0, tzinfo=IST)
    last_success = datetime(2026, 10, 8, 10, 59, tzinfo=IST)

    message = _format_health_alert(
        runner,
        market_data_status="OK",
        last_market_data_at=last_success,
        at=at,
    )

    assert "💓 HEALTH · EMA9" in message
    assert "Status: RUNNING" in message
    assert "Instrument: NIFTY" in message
    assert "Mode: PAPER" in message
    assert "Market data: OK (last success: 2026-10-08 10:59:00 IST)" in message
    assert "Position: FLAT" in message
    assert "Realized option P&L today: +₹125.50 (2 trade(s))" in message
    assert f"Daily loss cap: ₹{runner.daily_loss_cap:.2f} | not halted" in message


@pytest.mark.parametrize(
    ("at", "expected"),
    [
        (datetime(2026, 10, 26, 9, 14, tzinfo=IST), False),
        (datetime(2026, 10, 26, 9, 15, tzinfo=IST), True),
        (datetime(2026, 10, 26, 15, 19, tzinfo=IST), True),
        (datetime(2026, 10, 26, 15, 20, tzinfo=IST), False),
        (datetime(2026, 10, 25, 11, 0, tzinfo=IST), False),
    ],
)
def test_market_data_session_bounds(at, expected):
    assert _is_market_data_session(at) is expected


@pytest.mark.parametrize(
    ("today", "last_bar_time"),
    [
        (date(2026, 10, 26), "2026-10-26 15:14:00"),
        (date(2026, 10, 25), "2026-10-23 15:14:00"),
        (date(2026, 10, 26), "2026-10-22 15:14:00"),
    ],
)
def test_latest_trading_day_uses_latest_fyers_candle(today, last_bar_time):
    class HistoryApi:
        def history(self, symbol, *, resolution, start, end):
            assert symbol == "BSE:SENSEX-INDEX"
            assert resolution == "1"
            assert end == today
            return pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-10-22 15:14:00", last_bar_time],
                        errors="coerce",
                    ).tz_localize(IST)
                }
            )

    assert (
        _latest_trading_day(
            HistoryApi(),
            "BSE:SENSEX-INDEX",
            today=today,
        )
        == date.fromisoformat(last_bar_time[:10])
    )


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"EMA9_INSTRUMENT": "nifty"}, "NIFTY"),
        ({"EMA9_INSTRUMENT": "SENSEX"}, "SENSEX"),
    ],
)
def test_launcher_selects_exactly_one_instrument(environment, expected):
    assert selected_instrument(environment) == expected


@pytest.mark.parametrize("environment", [{}, {"EMA9_INSTRUMENT": "BOTH"}])
def test_launcher_rejects_missing_or_multiple_instruments(environment):
    with pytest.raises(ValueError, match="NIFTY or SENSEX"):
        selected_instrument(environment)


def test_nifty_selects_two_strikes_itm_in_nearest_upcoming_expiry():
    expiry = date(2026, 10, 8)
    rows = [
        {
            "underlying": "NIFTY",
            "option_type": option_type,
            "expiry": int(datetime(expiry.year, expiry.month, expiry.day, tzinfo=ZoneInfo("UTC")).timestamp()),
            "strike_price": strike,
            "lot_size": 65,
            "symbol": f"NSE:NIFTY-{option_type}-{strike}",
        }
        for option_type in ("CE", "PE")
        for strike in (24000, 24050, 24100, 24150, 24200)
    ] + [
        {
            "underlying": "NIFTY",
            "option_type": option_type,
            "expiry": int(datetime(2026, 10, 15, tzinfo=ZoneInfo("UTC")).timestamp()),
            "strike_price": strike,
            "lot_size": 65,
            "symbol": f"NSE:NIFTY-NEXT-{option_type}-{strike}",
        }
        for option_type in ("CE", "PE")
        for strike in (24000, 24050, 24100, 24150, 24200)
    ]
    config = InstrumentConfig(
        "NIFTY", "NIFTY", "NSE:NIFTY50-INDEX", "NSE", "NEAREST"
    )

    call = select_itm_contract(
        rows, action="ENTER_LONG", spot=24110, trade_day=date(2026, 10, 6), config=config
    )
    put = select_itm_contract(
        rows, action="ENTER_SHORT", spot=24110, trade_day=date(2026, 10, 6), config=config
    )

    assert call.expiry == expiry and call.strike == 24000 and call.option_type == "CE"
    assert put.expiry == expiry and put.strike == 24200 and put.option_type == "PE"


def test_sensex_uses_nearest_listed_expiry_and_bse_rows():
    rows = [
        {
            "underlying": "SENSEX",
            "option_type": "CE",
            "expiry": expiry.isoformat(),
            "strike_price": strike,
            "lot_size": 20,
            "symbol": f"BSE:SENSEX-{expiry}-{strike}-CE",
        }
        for expiry in (date(2026, 10, 8), date(2026, 10, 15))
        for strike in (83800, 83900, 84000, 84100, 84200)
    ]
    config = InstrumentConfig(
        "SENSEX", "SENSEX", "BSE:SENSEX-INDEX", "BSE", "NEAREST"
    )

    contract = select_itm_contract(
        rows,
        action="ENTER_LONG",
        spot=84110,
        trade_day=date(2026, 10, 8),
        config=config,
    )

    assert contract.expiry == date(2026, 10, 8)
    assert contract.strike == 83900


def test_quote_reader_rejects_missing_or_stale_quote_timestamps():
    class QuoteApi:
        def __init__(self, response):
            self.response = response

        def quotes(self, _symbols):
            return self.response

    now = datetime(2026, 10, 9, 10, 0, tzinfo=IST)
    stale = {
        "d": [
            {
                "n": "NSE:OPTION",
                "v": {"lp": 100, "tt": datetime(2026, 10, 9, 9, 58, tzinfo=IST).timestamp()},
            }
        ]
    }

    with pytest.raises(FyersQuoteError, match="quote_stale"):
        _quote_option_price(QuoteApi(stale), "NSE:OPTION", now=now)


@pytest.mark.parametrize(("side", "expected"), [("BUY", 101.0), ("SELL", 99.0)])
def test_quote_reader_uses_executable_order_book_side_when_ltp_is_stale(side, expected):
    class QuoteApi:
        def quotes(self, _symbols):
            return {
                "s": "ok",
                "d": [
                    {
                        "n": "NSE:OPTION",
                        "v": {
                            "lp": 100,
                            "tt": datetime(2026, 10, 9, 9, 58, tzinfo=IST).timestamp(),
                            "bid": 99,
                            "ask": 101,
                        },
                    }
                ],
            }

    price = _quote_option_price(
        QuoteApi(),
        "NSE:OPTION",
        now=datetime(2026, 10, 9, 10, 0, tzinfo=IST),
        side=side,
    )

    assert price == expected


def test_quote_reader_rejects_stale_ltp_with_invalid_order_book():
    class QuoteApi:
        def quotes(self, _symbols):
            return {
                "d": [
                    {
                        "n": "NSE:OPTION",
                        "v": {
                            "lp": 100,
                            "tt": datetime(2026, 10, 9, 9, 58, tzinfo=IST).timestamp(),
                            "bid": 101,
                            "ask": 99,
                        },
                    }
                ]
            }

    with pytest.raises(FyersQuoteError, match="order_book_invalid"):
        _quote_option_price(
            QuoteApi(),
            "NSE:OPTION",
            now=datetime(2026, 10, 9, 10, 0, tzinfo=IST),
        )


def test_quote_reader_logs_when_requested_symbol_is_not_returned(caplog):
    class QuoteApi:
        def quotes(self, _symbols):
            return {
                "s": "ok",
                "d": [
                    {
                        "n": "BSE:OTHER-OPTION",
                        "v": {"lp": 100, "tt": datetime(2026, 10, 9, 9, 59, tzinfo=IST).timestamp()},
                    }
                ],
            }

    now = datetime(2026, 10, 9, 10, 0, tzinfo=IST)
    with pytest.raises(FyersQuoteError, match="symbol_not_returned"):
        _quote_option_price(QuoteApi(), "BSE:SENSEX-OPTION", now=now)

    assert "reason=symbol_not_returned" in caplog.text
    assert "BSE:OTHER-OPTION" in caplog.text


def test_quote_reader_logs_fyers_request_failure(caplog):
    class QuoteApi:
        def quotes(self, _symbols):
            raise FyersAPIError("Fyers /quotes returned HTTP 429.")

    with pytest.raises(FyersQuoteError, match="HTTP 429"):
        _quote_option_price(
            QuoteApi(),
            "NSE:NIFTY-OPTION",
            now=datetime(2026, 10, 9, 10, 0, tzinfo=IST),
        )

    assert "EMA9 option quote request failed" in caplog.text
    assert "HTTP 429" in caplog.text


def test_quote_reader_returns_fresh_ltp():
    class QuoteApi:
        def quotes(self, _symbols):
            return {
                "d": [
                    {
                        "n": "NSE:OPTION",
                        "v": {
                            "lp": 101.25,
                            "tt": datetime(2026, 10, 9, 9, 59, 40, tzinfo=IST).timestamp(),
                        },
                    }
                ]
            }

    assert _quote_option_price(
        QuoteApi(), "NSE:OPTION", now=datetime(2026, 10, 9, 10, 0, tzinfo=IST)
    ) == 101.25


class _FakeApi:
    def __init__(self, quote_price=100.0, quote_timestamp=None):
        self.quote_price = quote_price
        self.quote_timestamp = quote_timestamp

    def quotes(self, symbols):
        quote_timestamp = self.quote_timestamp or datetime.now(IST).timestamp()
        return {
            "d": [
                {
                    "n": symbols[0],
                    "v": {"lp": self.quote_price, "tt": quote_timestamp},
                }
            ]
        }

    def option_master(self, _exchange):
        expiry = datetime(2026, 10, 15, tzinfo=ZoneInfo("UTC")).timestamp()
        return [
            {
                "underlying": "NIFTY",
                "option_type": option_type,
                "expiry": expiry,
                "strike_price": strike,
                "lot_size": 1,
                "symbol": f"NSE:NIFTY-{option_type}-{strike}",
            }
            for option_type in ("CE", "PE")
            for strike in (900, 950, 1000, 1050, 1100)
        ]


class _NoOrderExecution:
    def __init__(self):
        self.orders = []

    def place_market_order(self, *args, **kwargs):
        self.orders.append((args, kwargs))
        raise AssertionError("PAPER mode must not submit broker orders.")


class _RecordingNotifier:
    def __init__(self):
        self.messages = []
        self.errors = []

    def notify(self, message):
        self.messages.append(message)
        return True

    def notify_error(self, key, message):
        self.errors.append((key, message))
        return True

    def close(self, *, timeout):
        return None


class _SignalLogic:
    class EMATrendConfig:
        pass

    class EMATrendSignalEngine:
        def __init__(self, _config):
            pass

        def evaluate_candle(self, _frame):
            return type("Decision", (), {"action": "ENTER_LONG"})()

    @staticmethod
    def build_ema_trend_with_indicators(bars, _config):
        return bars.copy()


def _forward_runner(
    tmp_path, api, execution, *, mode="PAPER", notifier=None, dashboard_state=None
):
    return EMA9ForwardRunner(
        api=api,
        execution=execution,
        config=InstrumentConfig(
            "NIFTY", "NIFTY", "NSE:NIFTY50-INDEX", "NSE", "NEAREST"
        ),
        mode=mode,
        state_path=tmp_path / "state.json",
        trades_path=tmp_path / "trades.csv",
        notifier=notifier,
        dashboard_state=dashboard_state,
        signal_logic=_SignalLogic,
        strategy_config=_SignalLogic.EMATrendConfig(),
        daily_loss_cap=80,
    )


def test_dashboard_snapshot_reflects_open_option_mark_and_pnl(tmp_path):
    dashboard_state = FyersDashboardState()
    runner = _forward_runner(
        tmp_path,
        _FakeApi(),
        _NoOrderExecution(),
        dashboard_state=dashboard_state,
    )
    now = datetime(2026, 10, 8, 10, 0, tzinfo=IST)
    runner.daily_realized_option_pnl = 25
    runner.position = ForwardPosition(
        trade_id="dashboard-trade",
        signal="ENTER_LONG",
        signal_bar_start=now.isoformat(),
        entry_time=now.isoformat(),
        entry_index=1000,
        trailing_stop=997,
        contract=FyersOptionContract(
            "NSE:NIFTY-CE-1000", now.date() + timedelta(days=7), "CE", 1000, 1
        ),
        quantity=2,
        entry_option_price=100,
    )

    runner._update_dashboard_position(mark_price=110, open_pnl=20)
    snapshot = dashboard_state.snapshot()

    assert snapshot["position"]["mark_price"] == 110
    assert snapshot["position"]["open_pnl"] == 20
    assert snapshot["account"]["realized_pnl"] == 25
    assert snapshot["account"]["total_pnl"] == 45


def test_paper_entry_uses_itm_contract_without_submitting_an_order(tmp_path):
    now = datetime(2026, 10, 8, 10, 6, tzinfo=IST)
    api = _FakeApi(quote_timestamp=now.timestamp())
    execution = _NoOrderExecution()
    notifier = _RecordingNotifier()
    runner = _forward_runner(tmp_path, api, execution, notifier=notifier)
    timestamps = [
        datetime(2026, 10, 8, 9, 15, tzinfo=IST) + timedelta(minutes=index)
        for index in range(50)
    ]
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1000.0] * len(timestamps),
            "high": [1001.0] * len(timestamps),
            "low": [999.0] * len(timestamps),
            "close": [1000.0] * len(timestamps),
            "volume": [1.0] * len(timestamps),
        }
    )

    runner.process_market_data(bars, now=now)

    assert runner.position is not None
    assert runner.position.contract.option_type == "CE"
    assert runner.position.contract.strike == 900
    assert runner.position.entry_option_price == 100
    assert execution.orders == []
    assert len(notifier.messages) == 1
    assert "🚀 ENTRY · EMA" in notifier.messages[0]
    assert "Direction: LONG" in notifier.messages[0]
    assert "[PAPER]" in notifier.messages[0]


def test_paper_option_loss_cap_closes_position_and_halts_without_orders(tmp_path):
    api = _FakeApi(quote_price=10.0)
    execution = _NoOrderExecution()
    notifier = _RecordingNotifier()
    runner = _forward_runner(tmp_path, api, execution, notifier=notifier)
    now = datetime.now(IST)
    runner.trading_day = now.date()
    runner.position = ForwardPosition(
        trade_id="test-trade",
        signal="ENTER_LONG",
        signal_bar_start=now.isoformat(),
        entry_time=now.isoformat(),
        entry_index=1000,
        trailing_stop=997,
        contract=FyersOptionContract(
            "NSE:NIFTY-CE-1000", now.date() + timedelta(days=7), "CE", 1000, 1
        ),
        quantity=1,
        entry_option_price=100,
    )

    runner.process_market_data(pd.DataFrame(), now=now)

    assert runner.position is None
    assert runner.daily_loss_halted is True
    assert runner.daily_realized_option_pnl == -90
    assert execution.orders == []
    assert (tmp_path / "trades.csv").is_file()
    assert any("🏁 EXIT · EMA (" in message for message in notifier.messages)
    assert all("Index points:" not in message for message in notifier.messages)


def test_end_of_day_report_contains_trade_count_pnl_and_is_sent_once(tmp_path):
    notifier = _RecordingNotifier()
    runner = _forward_runner(
        tmp_path, _FakeApi(), _NoOrderExecution(), notifier=notifier
    )
    now = datetime(2026, 10, 8, 15, 15, tzinfo=IST)
    runner.trading_day = now.date()
    runner.daily_trade_count = 2
    runner.daily_realized_option_pnl = 125.5

    runner.send_end_of_day_report(now=now)
    runner.send_end_of_day_report(now=now)

    assert len(notifier.messages) == 1
    assert "📊 End-of-Day P&L [PAPER]" in notifier.messages[0]
    assert "NIFTY [PAPER]: +₹125.50 (2 trade(s))" in notifier.messages[0]
    assert runner.eod_report_sent is True


def test_overnight_cleanup_removes_tokens_and_strategy_caches(tmp_path, monkeypatch):
    class Session:
        closed = False

        def close(self):
            self.closed = True

    api_session = Session()
    api = FyersAPI("app-id", "old-token", session=api_session)
    runner = _forward_runner(tmp_path, api, _NoOrderExecution())
    runner._save_state()
    runner.trades_path.write_text("trade history\n", encoding="utf-8")
    token_file = tmp_path / "token"
    token_file.write_text("old-token", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("FYERS_ACCESS_TOKEN=old-token\nOTHER_SETTING=keep\n", encoding="utf-8")
    monkeypatch.setattr(
        "Dependencies.fyers_ema9_forward_runner.ENV_FILE", env_file
    )
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", "old-token")
    environment = {"FYERS_ACCESS_TOKEN": "old-token"}
    dashboard_state = FyersDashboardState()
    dashboard_state.update_market(candles=[{"time": 1}], spot=24000.0)
    auth_service = type("AuthService", (), {"token_file": token_file})()

    _clear_overnight_caches(
        api=api,
        auth_service=auth_service,
        runner=runner,
        dashboard_state=dashboard_state,
        environment=environment,
    )

    assert not token_file.exists()
    assert not runner.state_path.exists()
    assert not runner.trades_path.exists()
    assert env_file.read_text(encoding="utf-8") == "OTHER_SETTING=keep\n"
    assert "FYERS_ACCESS_TOKEN" not in os.environ
    assert "FYERS_ACCESS_TOKEN" not in environment
    assert api.access_token == ""
    assert api._requests_enabled is False
    assert api_session.closed
    assert dashboard_state.snapshot()["market"]["candles"] == []


def test_minute_touch_exit_uses_prior_completed_ema9_before_five_minute_close(tmp_path):
    now = datetime(2026, 10, 8, 10, 7, tzinfo=IST)
    api = _FakeApi(quote_timestamp=now.timestamp())
    execution = _NoOrderExecution()
    dashboard_state = FyersDashboardState()
    runner = _forward_runner(
        tmp_path, api, execution, dashboard_state=dashboard_state
    )
    runner.trading_day = now.date()
    runner.last_processed_minute = pd.Timestamp(datetime(2026, 10, 8, 10, 4, tzinfo=IST))
    runner.last_processed_bar = pd.Timestamp(datetime(2026, 10, 8, 10, 0, tzinfo=IST))
    runner.position = ForwardPosition(
        trade_id="touch-trade",
        signal="ENTER_LONG",
        signal_bar_start="2026-10-08T09:55:00+05:30",
        entry_time="2026-10-08T10:05:00+05:30",
        entry_index=1000,
        trailing_stop=997,
        contract=FyersOptionContract(
            "NSE:NIFTY-CE-1000", date(2026, 10, 15), "CE", 1000, 1
        ),
        quantity=1,
        entry_option_price=100,
    )
    timestamps = [
        datetime(2026, 10, 8, 9, 15, tzinfo=IST) + timedelta(minutes=index)
        for index in range(52)
    ]
    lows = [999.0] * len(timestamps)
    lows[50] = 996.0
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1000.0] * len(timestamps),
            "high": [1001.0] * len(timestamps),
            "low": lows,
            "close": [1000.0] * len(timestamps),
            "volume": [1.0] * len(timestamps),
        }
    )

    runner.process_market_data(bars, now=now)

    assert runner.position is None
    assert "EMA9_TOUCH" in (tmp_path / "trades.csv").read_text()
    assert execution.orders == []
    assert dashboard_state.snapshot()["trades"][0]["direction"] == "LONG"


def test_live_runner_closes_when_option_quote_becomes_stale(tmp_path):
    class StaleApi:
        def quotes(self, _symbols):
            old = now - timedelta(minutes=2)
            return {
                "d": [
                    {
                        "n": "NSE:NIFTY-CE-1000",
                        "v": {"lp": 90, "tt": old.timestamp()},
                    }
                ]
            }

    class FilledExecution:
        def __init__(self):
            self.orders = []

        def place_market_order(self, *args, **kwargs):
            self.orders.append((args, kwargs))
            return type(
                "Result",
                (),
                {"status": OrderStatus.FILLED, "average_fill_price": 88.0},
            )()

    execution = FilledExecution()
    now = datetime.now(IST).replace(hour=10, minute=0, second=0, microsecond=0)
    runner = _forward_runner(tmp_path, StaleApi(), execution, mode="LIVE")
    runner.trading_day = now.date()
    runner.position = ForwardPosition(
        trade_id="live-stale-trade",
        signal="ENTER_LONG",
        signal_bar_start=now.isoformat(),
        entry_time=now.isoformat(),
        entry_index=1000,
        trailing_stop=997,
        contract=FyersOptionContract(
            "NSE:NIFTY-CE-1000", now.date() + timedelta(days=7), "CE", 1000, 1
        ),
        quantity=1,
        entry_option_price=100,
    )

    runner.check_open_position_risk(now=now)

    assert runner.position is None
    assert len(execution.orders) == 1
    assert execution.orders[0][0][1] == "SELL"
    assert "STALE_OPTION_QUOTE" in (tmp_path / "trades.csv").read_text()


def test_live_runner_refuses_to_idle_with_a_position_from_another_session(tmp_path):
    runner = _forward_runner(tmp_path, _FakeApi(), _NoOrderExecution(), mode="LIVE")
    now = datetime(2026, 10, 9, 10, 0, tzinfo=IST)
    runner.trading_day = date(2026, 10, 8)
    runner.position = ForwardPosition(
        trade_id="prior-day",
        signal="ENTER_LONG",
        signal_bar_start="2026-10-08T10:00:00+05:30",
        entry_time="2026-10-08T10:05:00+05:30",
        entry_index=1000,
        trailing_stop=997,
        contract=FyersOptionContract(
            "NSE:NIFTY-CE-1000", date(2026, 10, 15), "CE", 1000, 1
        ),
        quantity=1,
        entry_option_price=100,
    )

    with pytest.raises(RuntimeError, match="outside its trading session"):
        _validate_live_resume(runner, now=now)
