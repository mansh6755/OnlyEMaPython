from datetime import date

import pandas as pd

from Dependencies.fyers_ema_replay import _action_polls, _quote_at, replay


class _FakeAPI:
    def __init__(self, option_bars):
        self.option_bars = option_bars
        self.requests = []

    def history(self, symbol, **kwargs):
        self.requests.append((symbol, kwargs))
        return self.option_bars.copy()


class _FakeLogic:
    class EMATrendPositionContext:
        def __init__(self, direction, entry_underlying, stop_underlying):
            self.direction = direction
            self.entry_underlying = entry_underlying
            self.stop_underlying = stop_underlying

    class EMATrendDecision:
        def __init__(self, action="HOLD", exit_reason="", signal_triggered=False):
            self.action = action
            self.exit_reason = exit_reason
            self.signal_triggered = signal_triggered
            self.stop_underlying = 0

    class EMATrendSignalEngine:
        def __init__(self, _config):
            pass

        def evaluate_candle(self, candles, position=None):
            timestamp = pd.Timestamp(candles.iloc[-1]["timestamp"])
            if position is None and timestamp == pd.Timestamp("2026-10-15T09:25:00+05:30"):
                return _FakeLogic.EMATrendDecision("ENTER_LONG", signal_triggered=True)
            if position is not None and timestamp == pd.Timestamp("2026-10-15T09:30:00+05:30"):
                return _FakeLogic.EMATrendDecision("EXIT", exit_reason="EMA11_EXIT")
            return _FakeLogic.EMATrendDecision()

    @staticmethod
    def build_ema_trend_with_indicators(bars, _config):
        return bars.copy()


def test_quote_at_uses_only_an_option_candle_completed_by_the_requested_time():
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-10-15T09:30:00+05:30", "2026-10-15T09:30:05+05:30"]),
            "close": [100.0, 105.0],
        }
    )

    quote = _quote_at(
        bars,
        pd.Timestamp("2026-10-15T09:30:07+05:30"),
        "5S",
    )

    assert quote == (100.0, pd.Timestamp("2026-10-15T09:30:05+05:30"))


def test_poll_schedule_includes_the_first_poll_after_square_off():
    polls = _action_polls(date(2026, 10, 15), poll_seconds=45)

    assert polls[-1] >= pd.Timestamp("2026-10-15T15:15:00+05:30")
    assert polls[-1] < pd.Timestamp("2026-10-15T15:16:00+05:30")


def test_replay_uses_live_poll_timing_and_option_prices_for_trade_pnl():
    day = date(2026, 10, 15)
    timestamps = pd.date_range(
        "2026-10-15T09:15:00+05:30",
        "2026-10-15T15:15:00+05:30",
        freq="5s",
    )
    option_bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 110.0,
            "low": 99.0,
            "close": [100.0 if stamp < pd.Timestamp("2026-10-15T09:34:55+05:30") else 110.0 for stamp in timestamps],
            "volume": 1,
        }
    )
    index_bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-10-15T09:25:00+05:30", "2026-10-15T09:30:00+05:30"]),
            "open": [22500.0, 22510.0],
            "high": [22520.0, 22520.0],
            "low": [22490.0, 22500.0],
            "close": [22510.0, 22505.0],
            "volume": [5, 5],
        }
    )
    api = _FakeAPI(option_bars)
    active_rows = [
        {
            "underlying": "NIFTY",
            "symbol": "NSE:NIFTY26O2022500CE",
            "expiry": date(2026, 10, 20),
            "strike_price": 22500.0,
            "option_type": "CE",
            "lot_size": 65,
        }
    ]

    trades, counters = replay(
        signal_logic=_FakeLogic,
        strategy_config=None,
        index_bars=index_bars,
        api=api,
        active_rows=active_rows,
        expiries=[date(2026, 10, 20)],
        start=day,
        end=day,
    )

    assert counters == {"signals": 1, "entries": 1, "exits": 1, "daily_loss_halts": 0}
    assert trades.iloc[0]["signal_time"] == "2026-10-15T09:30:00+05:30"
    assert trades.iloc[0]["signal_bar_start"] == "2026-10-15T09:25:00+05:30"
    assert trades.iloc[0]["entry_time"] == "2026-10-15T09:30:00+05:30"
    assert trades.iloc[0]["entry_quote_time"] == "2026-10-15T09:30:00+05:30"
    assert trades.iloc[0]["exit_time"] == "2026-10-15T09:35:00+05:30"
    assert trades.iloc[0]["exit_quote_time"] == "2026-10-15T09:35:00+05:30"
    assert trades.iloc[0]["entry_option"] == 100.0
    assert trades.iloc[0]["exit_option"] == 110.0
    assert trades.iloc[0]["quantity"] == 65
    assert trades.iloc[0]["gross_pnl"] == 650.0
    assert trades.iloc[0]["exit_reason"] == "EMA11_EXIT"
    assert api.requests[0][0] == "NSE:NIFTY26O2022500CE"
