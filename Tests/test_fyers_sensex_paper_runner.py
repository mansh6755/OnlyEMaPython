import json
from collections import deque
from datetime import date

import pandas as pd
import pytest

from Dependencies import fyers_sensex_paper_runner as paper_runner
from Dependencies.fyers_options import FyersOptionContract
from Dependencies.fyers_sensex_paper_runner import (
    PaperPosition,
    SensexPaperEngine,
    _format_exit_message,
    _require_paper_mode,
)


class _FakeSignalLogic:
    class EMATrendSignalEngine:
        def __init__(self, _config):
            pass

        def evaluate_candle(self, candles):
            signal_time = pd.Timestamp(candles.iloc[-1]["timestamp"])
            action = (
                "ENTER_LONG"
                if signal_time == pd.Timestamp("2026-01-01T09:20:00+05:30")
                else "HOLD"
            )
            return type("Decision", (), {"action": action})()


class _FakeAPI:
    contract = FyersOptionContract(
        "BSE:SENSEX26JAN80000CE",
        date(2026, 1, 8),
        "CE",
        80000.0,
        20,
    )

    def __init__(self, option_price=100.0):
        self.option_price = option_price
        self.quote_prices = deque()
        self.resolved = []
        self.quoted = []

    def resolve_nearest_option(self, action, *, spot, today):
        self.resolved.append((action, spot, today))
        return self.contract

    def quote_ltp(self, symbol):
        self.quoted.append(symbol)
        return self.quote_prices.popleft() if self.quote_prices else self.option_price


class _FakeNotifier:
    def __init__(self):
        self.messages = []

    def notify(self, message):
        self.messages.append(message)
        return True


def _bars(*, close=600.0, ema9=900.0):
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T09:15:00+05:30",
                    "2026-01-01T09:20:00+05:30",
                    "2026-01-01T09:25:00+05:30",
                ]
            ),
            "open": [1000.0, 1000.0, 1000.0],
            "high": [1000.0, 1001.0, 1001.0],
            "low": [1000.0, 999.0, close - 10.0],
            "close": [1000.0, 1000.0, close],
            "ema9": [ema9, ema9, ema9],
        }
    )


def _minutes(*, first_low=999.0, first_open=1000.0):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01T09:25:00+05:30",
                periods=5,
                freq="min",
            ),
            "open": [first_open, 1000.0, 1000.0, 1000.0, 1000.0],
            "high": [1001.0, 1001.0, 1001.0, 1001.0, 1001.0],
            "low": [first_low, 999.0, 999.0, 999.0, 590.0],
            "close": [1000.0, 1000.0, 1000.0, 1000.0, 600.0],
        }
    )


def _engine(tmp_path, cap=3500, ema9_buffer_points=0.0):
    return SensexPaperEngine(
        daily_option_loss_cap=cap,
        ema9_buffer_points=ema9_buffer_points,
        trades_path=tmp_path / "trades.csv",
        state_path=tmp_path / "state.json",
    )


def _enter_position(engine, api, notifier, *, ema9=900.0):
    engine.initialize(pd.Timestamp("2026-01-01T09:15:00+05:30"), date(2026, 1, 1))
    engine.process_completed_bars(
        _bars(ema9=ema9).iloc[:2],
        _minutes().iloc[:0],
        start=date(2026, 1, 1),
        end=date(2026, 1, 1),
        now=pd.Timestamp("2026-01-01T09:26:00+05:30").to_pydatetime(),
        signal_logic=_FakeSignalLogic,
        strategy_config=object(),
        api=api,
        notifier=notifier,
    )


def test_paper_entry_resolves_atm_call_and_uses_contract_lot_and_option_ltp(tmp_path):
    engine = _engine(tmp_path)
    api = _FakeAPI(option_price=177.70)
    notifier = _FakeNotifier()

    _enter_position(engine, api, notifier)

    assert engine.position is not None
    assert engine.position.entry_index == 1000.0
    assert engine.position.entry_time == "2026-01-01T09:25:00+05:30"
    assert engine.position.contract == api.contract
    assert engine.position.quantity == 20
    assert engine.position.entry_option_price == 177.70
    assert api.resolved == [("ENTER_LONG", 1000.0, date(2026, 1, 1))]
    assert notifier.messages[0].startswith("🚀 ENTRY · EMA\nDirection: LONG")
    assert "Instrument: BUY SENSEX-Jan2026-80000-CE" in notifier.messages[0]
    assert "Entry: ₹177.70" in notifier.messages[0]


def test_paper_entry_applies_configured_ema9_buffer(tmp_path):
    engine = _engine(tmp_path, ema9_buffer_points=3.0)
    api = _FakeAPI()
    notifier = _FakeNotifier()

    _enter_position(engine, api, notifier, ema9=900.0)

    assert engine.position is not None
    assert engine.position.trailing_stop == 897.0
    state = json.loads(engine.state_path.read_text(encoding="utf-8"))
    assert state["ema9_buffer_points"] == 3.0


def test_paper_minute_ohlc_touch_uses_prior_ema_and_option_ltp_for_rupee_pnl(tmp_path):
    engine = _engine(tmp_path)
    api = _FakeAPI(option_price=177.70)
    api.quote_prices.extend([177.70, 139.50])
    notifier = _FakeNotifier()
    _enter_position(engine, api, notifier)

    engine.process_completed_bars(
        _bars(close=1000.0),
        _minutes(first_low=899.0),
        start=date(2026, 1, 1),
        end=date(2026, 1, 1),
        now=pd.Timestamp("2026-01-01T09:30:00+05:30").to_pydatetime(),
        signal_logic=_FakeSignalLogic,
        strategy_config=object(),
        api=api,
        notifier=notifier,
    )

    trade = pd.read_csv(engine.trades_path).iloc[0]
    assert trade["exit_reason"] == "EMA9_TOUCH"
    assert trade["exit_index"] == 900.0
    assert trade["index_points"] == -100.0
    assert trade["entry_option_price"] == 177.70
    assert trade["exit_option_price"] == 139.50
    assert trade["option_pnl"] == pytest.approx(-764.0)
    assert trade["daily_option_pnl"] == pytest.approx(-764.0)
    assert trade["daily_option_loss_cap"] == 3500.0
    assert engine.daily_realized_option_pnl == pytest.approx(-764.0)
    assert trade["exit_time"] == "2026-01-01T09:26:00+05:30"
    assert "SELL SENSEX-Jan2026-80000-CE" in notifier.messages[-1]
    assert "-₹764.00" in notifier.messages[-1]


def test_paper_minute_touch_uses_buffered_prior_ema9(tmp_path):
    engine = _engine(tmp_path, ema9_buffer_points=3.0)
    api = _FakeAPI(option_price=177.70)
    api.quote_prices.extend([177.70, 139.50])
    notifier = _FakeNotifier()
    _enter_position(engine, api, notifier, ema9=900.0)

    engine.process_completed_bars(
        _bars(close=1000.0),
        _minutes(first_low=896.0),
        start=date(2026, 1, 1),
        end=date(2026, 1, 1),
        now=pd.Timestamp("2026-01-01T09:30:00+05:30").to_pydatetime(),
        signal_logic=_FakeSignalLogic,
        strategy_config=object(),
        api=api,
        notifier=notifier,
    )

    trade = pd.read_csv(engine.trades_path).iloc[0]
    assert trade["exit_reason"] == "EMA9_TOUCH"
    assert trade["exit_index"] == 897.0


def test_index_points_loss_alone_does_not_trigger_option_rupee_cap(tmp_path):
    engine = _engine(tmp_path, ema9_buffer_points=3.0)
    api = _FakeAPI(option_price=177.70)
    api.quote_prices.extend([177.70, 139.50])
    notifier = _FakeNotifier()
    _enter_position(engine, api, notifier, ema9=500.0)
    minute_bars = _minutes()
    minute_bars["open"] = [800.0, 750.0, 700.0, 700.0, 700.0]
    minute_bars["high"] = [805.0] * 5
    minute_bars["low"] = [795.0, 695.0, 695.0, 695.0, 695.0]
    minute_bars["close"] = [800.0, 700.0, 700.0, 700.0, 700.0]

    engine.process_completed_bars(
        _bars(close=1000.0, ema9=500.0),
        minute_bars,
        start=date(2026, 1, 1),
        end=date(2026, 1, 1),
        now=pd.Timestamp("2026-01-01T09:30:00+05:30").to_pydatetime(),
        signal_logic=_FakeSignalLogic,
        strategy_config=object(),
        api=api,
        notifier=notifier,
    )

    assert engine.position is not None
    assert not engine.daily_loss_halted
    assert not engine.trades_path.exists()


def test_minute_ohlc_gap_through_fills_at_minute_open_and_ignores_incomplete_minute(tmp_path):
    engine = _engine(tmp_path)
    api = _FakeAPI(option_price=177.70)
    api.quote_prices.extend([177.70, 180.0])
    notifier = _FakeNotifier()
    _enter_position(engine, api, notifier)
    minutes = _minutes(first_low=850.0, first_open=875.0)
    now = pd.Timestamp("2026-01-01T09:30:00+05:30").to_pydatetime()

    engine.process_completed_bars(
        _bars(close=1000.0),
        minutes,
        start=date(2026, 1, 1),
        end=date(2026, 1, 1),
        now=now,
        signal_logic=_FakeSignalLogic,
        strategy_config=object(),
        api=api,
        notifier=notifier,
    )

    trade = pd.read_csv(engine.trades_path).iloc[0]
    assert trade["exit_index"] == 875.0
    assert trade["exit_time"] == "2026-01-01T09:26:00+05:30"


def test_option_rupee_cap_exits_open_position_and_halts_entries(tmp_path):
    engine = _engine(tmp_path, cap=700.0)
    api = _FakeAPI(option_price=177.70)
    notifier = _FakeNotifier()
    _enter_position(engine, api, notifier, ema9=500.0)
    api.option_price = 139.50

    trade = engine.close_if_option_loss_cap_breached(
        pd.Timestamp("2026-01-01T09:27:05+05:30"),
        990.0,
        api.option_price,
    )

    assert trade is not None
    assert trade["exit_reason"] == "OPTION_DAILY_MAX_LOSS"
    assert trade["exit_time"] == "2026-01-01T09:27:05+05:30"
    assert trade["index_points"] == -10.0
    assert trade["option_pnl"] == pytest.approx(-764.0)
    assert engine.daily_realized_option_pnl == pytest.approx(-764.0)
    assert engine.daily_loss_halted
    assert engine.position is None


def test_closed_trade_option_loss_reaching_cap_blocks_new_entries(tmp_path):
    engine = _engine(tmp_path, cap=700.0)
    api = _FakeAPI(option_price=177.70)
    notifier = _FakeNotifier()
    _enter_position(engine, api, notifier)
    api.option_price = 139.50

    trade = engine.close_position(
        pd.Timestamp("2026-01-01T09:27:00+05:30"),
        990.0,
        139.50,
        "EMA9_TOUCH",
    )

    assert trade["option_pnl"] == pytest.approx(-764.0)
    assert engine.daily_loss_halted
    assert engine.daily_realized_option_pnl == pytest.approx(-764.0)


def test_paper_state_restores_contract_and_rejects_changed_cap(tmp_path):
    engine = _engine(tmp_path, cap=250)
    api = _FakeAPI()
    notifier = _FakeNotifier()
    _enter_position(engine, api, notifier)

    restored = SensexPaperEngine(
        daily_option_loss_cap=250,
        ema9_buffer_points=0.0,
        trades_path=tmp_path / "trades.csv",
        state_path=tmp_path / "state.json",
    )
    assert restored.position == engine.position
    assert restored.last_processed_bar == engine.last_processed_bar

    with pytest.raises(RuntimeError, match="cap differs"):
        SensexPaperEngine(
            daily_option_loss_cap=300,
            ema9_buffer_points=0.0,
            trades_path=tmp_path / "trades.csv",
            state_path=tmp_path / "state.json",
        )


def test_paper_state_rejects_changed_ema9_buffer(tmp_path):
    engine = _engine(tmp_path, ema9_buffer_points=3.0)
    engine.save_state()

    with pytest.raises(RuntimeError, match="EMA9 buffer differs"):
        SensexPaperEngine(
            daily_option_loss_cap=3500,
            ema9_buffer_points=6.0,
            trades_path=tmp_path / "trades.csv",
            state_path=tmp_path / "state.json",
        )


def test_state_file_contains_no_option_order_execution_fields(tmp_path):
    engine = _engine(tmp_path)
    engine.save_state()

    state = json.loads(engine.state_path.read_text(encoding="utf-8"))
    assert state["version"] == 4
    assert state["daily_option_loss_cap"] == 3500
    assert state["daily_realized_option_pnl"] == 0.0
    assert "orders" not in state


def test_exit_ledger_deduplicates_after_interrupted_state_persist(tmp_path):
    engine = _engine(tmp_path)
    api = _FakeAPI(option_price=177.70)
    notifier = _FakeNotifier()
    _enter_position(engine, api, notifier)
    position = engine.position
    assert position is not None

    trade = engine.close_position(
        pd.Timestamp("2026-01-01T09:27:00+05:30"),
        900.0,
        139.50,
        "EMA9_TOUCH",
    )
    engine.position = position
    engine.close_position(
        pd.Timestamp("2026-01-01T09:28:00+05:30"),
        900.0,
        140.0,
        "EMA9_TOUCH",
    )

    ledger = pd.read_csv(engine.trades_path)
    assert trade["trade_id"] == ledger.loc[0, "trade_id"]
    assert len(ledger) == 1


@pytest.mark.parametrize("value", ["true", "True", "1", ""])
def test_paper_runner_refuses_non_false_live_flag(monkeypatch, value):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", value)

    with pytest.raises(RuntimeError, match="requires LIVE_TRADING_ENABLED=false"):
        _require_paper_mode()


def test_paper_runner_accepts_explicit_false_live_flag(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")

    _require_paper_mode()


def test_fyers_authentication_uses_telegram_when_no_saved_token(monkeypatch, tmp_path):
    auth_service = type(
        "FakeAuthService",
        (),
        {
            "start": lambda self: None,
            "authenticate_startup": lambda self, *, has_access_token: setattr(
                self, "has_access_token", has_access_token
            ),
            "stop": lambda self: None,
        },
    )()
    api_instances = []
    auth_configs = []

    def fake_api(app_id, token):
        api = type("FakeAPI", (), {})()
        api.app_id = app_id
        api.access_token = token
        api_instances.append(api)
        return api

    def fake_auth_service(**kwargs):
        auth_configs.append(kwargs)
        return auth_service

    monkeypatch.setenv("FYERS_APP_ID", "paper-app")
    monkeypatch.setenv("FYERS_TOKEN_FILE", str(tmp_path / "token"))
    monkeypatch.setenv("FYERS_SECRET_KEY", "app-secret")
    monkeypatch.setenv("FYERS_REDIRECT_URI", "https://paper.example/auth/callback")
    monkeypatch.setenv("FYERS_AUTH_TELEGRAM_CHAT_ID", "12345")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setattr(paper_runner, "load_persisted_access_token", lambda *_args: "")
    monkeypatch.setattr(paper_runner, "FyersAPI", fake_api)
    monkeypatch.setattr(paper_runner, "FyersTelegramAuthService", fake_auth_service)

    api, service = paper_runner._authenticate_fyers_through_telegram()

    assert api is api_instances[0]
    assert api.access_token == "telegram-auth-pending"
    assert service is auth_service
    assert service.has_access_token is False
    assert auth_configs[0]["port"] == 8080
    assert auth_configs[0]["authorized_chat_id"] == "12345"
    assert auth_configs[0]["redirect_uri"] == "https://paper.example/auth/callback"


def test_exit_message_includes_contract_option_pnl_and_index_points():
    position = PaperPosition(
        signal="ENTER_SHORT",
        signal_bar_start="2026-01-01T09:20:00+05:30",
        entry_time="2026-01-01T09:25:00+05:30",
        entry_index=1000.0,
        trailing_stop=900.0,
        contract=FyersOptionContract(
            "BSE:SENSEX26JAN80000PE",
            date(2026, 1, 8),
            "PE",
            80000.0,
            20,
        ),
        quantity=20,
        entry_option_price=100.0,
    )
    trade = {
        "exit_reason": "EMA9_TOUCH",
        "entry_option_price": 100.0,
        "exit_option_price": 125.0,
        "option_pnl": 500.0,
        "daily_option_pnl": 500.0,
        "daily_option_loss_cap": 3500.0,
        "index_points": 25.0,
        "exit_time": "2026-01-01T10:00:00+05:30",
    }

    message = _format_exit_message(position, trade)

    assert "Direction: SHORT" in message
    assert "SELL SENSEX-Jan2026-80000-PE" in message
    assert "+₹500.00" in message
    assert "Index points: +25.00" in message
