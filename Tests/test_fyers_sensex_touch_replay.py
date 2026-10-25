import pandas as pd

from Dependencies.fyers_sensex_touch_replay import replay_sensex_touch_cap


class _FakeSignalLogic:
    class EMATrendDecision:
        def __init__(self, action):
            self.action = action

    class EMATrendSignalEngine:
        def __init__(self, _config):
            pass

        def evaluate_candle(self, candles):
            signal_time = pd.Timestamp(candles.iloc[-1]["timestamp"])
            action = "ENTER_LONG" if signal_time == pd.Timestamp("2026-01-01T09:20:00+05:30") else "HOLD"
            return _FakeSignalLogic.EMATrendDecision(action)


def test_daily_cap_exits_at_five_minute_close_before_final_minute_touch():
    bar_times = pd.to_datetime(
        [
            "2026-01-01T09:15:00+05:30",
            "2026-01-01T09:20:00+05:30",
            "2026-01-01T09:25:00+05:30",
        ]
    )
    bars = pd.DataFrame(
        {
            "timestamp": bar_times,
            "open": [1000.0, 1000.0, 1000.0],
            "high": [1000.0, 1000.0, 1005.0],
            "low": [1000.0, 1000.0, 590.0],
            "close": [1000.0, 1000.0, 600.0],
            "ema9": [100.0, 100.0, 100.0],
        }
    )
    minute_times = pd.date_range(
        "2026-01-01T09:25:00+05:30",
        periods=5,
        freq="min",
    )
    minute_bars = pd.DataFrame(
        {
            "timestamp": minute_times,
            "open": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            "high": [1005.0, 1005.0, 1005.0, 1005.0, 1005.0],
            "low": [995.0, 995.0, 995.0, 995.0, 590.0],
            "close": [1000.0, 1000.0, 1000.0, 1000.0, 600.0],
        }
    )

    trades = replay_sensex_touch_cap(
        bars,
        minute_bars,
        start=pd.Timestamp("2026-01-01").date(),
        end=pd.Timestamp("2026-01-01").date(),
        daily_loss_cap=300.0,
        signal_logic=_FakeSignalLogic,
        strategy_config=object(),
    )

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "DAILY_MAX_LOSS"
    assert trades.iloc[0]["exit_time"] == "2026-01-01T09:30:00+05:30"
    assert trades.iloc[0]["index_points"] == -400.0


def test_ema9_buffer_offsets_long_initial_and_trailing_stops():
    bar_times = pd.to_datetime(
        [
            "2026-01-01T09:15:00+05:30",
            "2026-01-01T09:20:00+05:30",
            "2026-01-01T09:25:00+05:30",
            "2026-01-01T09:30:00+05:30",
        ]
    )
    bars = pd.DataFrame(
        {
            "timestamp": bar_times,
            "open": [1000.0] * 4,
            "high": [1001.0] * 4,
            "low": [999.0] * 4,
            "close": [1000.0, 1000.0, 1001.0, 1000.0],
            "ema9": [99.0, 100.0, 101.0, 102.0],
        }
    )
    minute_times = pd.date_range("2026-01-01T09:25:00+05:30", periods=10, freq="min")
    minute_bars = pd.DataFrame(
        {
            "timestamp": minute_times,
            "open": [99.0] * 10,
            "high": [100.0] * 10,
            "low": [98.5] * 5 + [97.5] * 5,
            "close": [99.0] * 10,
        }
    )

    trades = replay_sensex_touch_cap(
        bars,
        minute_bars,
        start=pd.Timestamp("2026-01-01").date(),
        end=pd.Timestamp("2026-01-01").date(),
        daily_loss_cap=300.0,
        ema9_buffer_points=3.0,
        signal_logic=_FakeSignalLogic,
        strategy_config=object(),
    )

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "EMA9_TOUCH"
    assert trades.iloc[0]["exit_time"] == "2026-01-01T09:31:00+05:30"
    assert trades.iloc[0]["exit_index"] == 98.0
