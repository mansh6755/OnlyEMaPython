import argparse
from datetime import date, time

import pandas as pd
import pytest

from Dependencies.fyers_ema_replay import (
    FyersHistoryCache,
    _action_polls,
    _expiry_lookup_end,
    _index_history_warmup_start,
    _parse_cache_flag,
    _quote_at,
    replay,
    replay_index_only,
    _fetch_index_history,
)


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
                return _FakeLogic.EMATrendDecision("EXIT", exit_reason="EMA9_EXIT")
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


def test_expiry_lookup_includes_the_following_week_after_the_replay_end():
    assert _expiry_lookup_end(date(2026, 2, 28), date(2026, 10, 15)) == date(2026, 3, 7)
    assert _expiry_lookup_end(date(2026, 10, 15), date(2026, 10, 15)) == date(2026, 10, 14)


def test_index_history_warmup_stays_inside_the_one_year_intraday_window():
    today = date(2026, 10, 23)

    assert _index_history_warmup_start(date(2025, 10, 23), today) == date(2025, 10, 23)
    assert _index_history_warmup_start(date(2025, 10, 30), today) == date(2025, 10, 23)
    assert _index_history_warmup_start(date(2025, 10, 22), today) == date(2025, 10, 15)


def test_use_cache_flag_accepts_true_or_false_values():
    assert _parse_cache_flag("true") is True
    assert _parse_cache_flag("FALSE") is False
    with pytest.raises(argparse.ArgumentTypeError, match="useCache must be true or false"):
        _parse_cache_flag("yes")


def test_history_cache_reuses_candles_across_backtest_runs(tmp_path):
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-10-15T09:15:00+05:30"]),
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [104.0],
            "volume": [10.0],
        }
    )
    api = _FakeAPI(bars)
    cache = FyersHistoryCache(tmp_path)
    request = {
        "resolution": "1",
        "start": date(2026, 10, 15),
        "end": date(2026, 10, 15),
    }

    first = cache.history(api, "NSE:NIFTY50-INDEX", **request)
    second = cache.history(api, "NSE:NIFTY50-INDEX", **request)

    assert len(api.requests) == 1
    assert cache.hits == 1
    assert cache.misses == 1
    pd.testing.assert_frame_equal(first, second)


def test_history_cache_separates_expired_and_active_history(tmp_path):
    bars = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    api = _FakeAPI(bars)
    cache = FyersHistoryCache(tmp_path)
    request = {
        "resolution": "1",
        "start": date(2026, 10, 15),
        "end": date(2026, 10, 15),
    }

    cache.history(api, "NSE:NIFTY26O1922750CE", expired=False, **request)
    cache.history(api, "NSE:NIFTY26O1922750CE", expired=True, **request)
    cache.history(api, "NSE:NIFTY26O1922750CE", expired=True, **request)

    assert len(api.requests) == 2
    assert cache.hits == 1
    assert cache.misses == 2


def test_history_cache_key_includes_symbol_resolution_and_date_range(tmp_path):
    bars = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    api = _FakeAPI(bars)
    cache = FyersHistoryCache(tmp_path)
    start = date(2026, 10, 15)
    end = date(2026, 10, 16)

    cache.history(api, "NSE:NIFTY50-INDEX", resolution="1", start=start, end=end)
    cache.history(api, "NSE:NIFTY50-INDEX", resolution="5S", start=start, end=end)
    cache.history(api, "NSE:NIFTY26O1922750CE", resolution="1", start=start, end=end)
    cache.history(api, "NSE:NIFTY50-INDEX", resolution="1", start=start, end=start)

    assert len(api.requests) == 4
    assert cache.hits == 0
    assert cache.misses == 4


def test_history_cache_can_be_disabled_without_creating_cache_files(tmp_path):
    bars = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    api = _FakeAPI(bars)
    cache = FyersHistoryCache(tmp_path, enabled=False)
    request = {
        "resolution": "1",
        "start": date(2026, 10, 15),
        "end": date(2026, 10, 15),
    }

    cache.history(api, "NSE:NIFTY50-INDEX", **request)
    cache.history(api, "NSE:NIFTY50-INDEX", **request)

    assert len(api.requests) == 2
    assert cache.misses == 2
    assert list(tmp_path.iterdir()) == []


def test_history_cache_reports_corrupt_files_instead_of_silently_refetching(tmp_path):
    api = _FakeAPI(pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]))
    cache = FyersHistoryCache(tmp_path)
    request = {
        "resolution": "1",
        "start": date(2026, 10, 15),
        "end": date(2026, 10, 15),
    }
    cache.history(api, "NSE:NIFTY50-INDEX", **request)
    path = next(tmp_path.glob("*.csv"))
    path.write_text("not,a,candle,file\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Clear that cache file"):
        cache.history(api, "NSE:NIFTY50-INDEX", **request)
    assert len(api.requests) == 1


def test_fetch_index_history_uses_one_day_requests_and_requested_symbol():
    bars = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    api = _FakeAPI(bars)

    result = _fetch_index_history(
        api,
        date(2026, 10, 19),
        date(2026, 10, 20),
        symbol="BSE:SENSEX-INDEX",
    )

    assert result.empty
    assert [request[0] for request in api.requests] == ["BSE:SENSEX-INDEX", "BSE:SENSEX-INDEX"]
    assert [request[1]["start"] for request in api.requests] == [date(2026, 10, 19), date(2026, 10, 20)]
    assert [request[1]["end"] for request in api.requests] == [date(2026, 10, 19), date(2026, 10, 20)]


def test_index_only_replay_reports_directional_points_without_option_data():
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-10-15T09:15:00+05:30", periods=5, freq="5min"),
            "open": [99.0, 99.0, 100.0, 101.0, 102.0],
            "high": [100.0, 101.0, 102.0, 103.0, 104.0],
            "low": [98.0, 98.0, 99.0, 100.0, 101.0],
            "close": [99.0, 100.0, 101.0, 102.0, 103.0],
            "volume": [10.0] * 5,
        }
    )

    trades, counters = replay_index_only(
        signal_logic=_FakeLogic,
        strategy_config=None,
        index_bars=bars,
        start=date(2026, 10, 15),
        end=date(2026, 10, 15),
        poll_seconds=30,
    )

    assert len(trades) == 1
    assert trades.iloc[0]["signal"] == "ENTER_LONG"
    assert trades.iloc[0]["entry_index"] == 101.0
    assert trades.iloc[0]["exit_index"] == 102.0
    assert trades.iloc[0]["index_points"] == 1.0
    assert trades.iloc[0]["exit_reason"] == "EMA9_EXIT"
    assert counters == {"signals": 1, "entries": 1, "exits": 1, "daily_loss_halts": 0}


@pytest.mark.parametrize(("max_daily_loss_points", "expected_loss"), [(130, -130), (None, -170)])
def test_index_only_daily_loss_cap_closes_position_and_blocks_day_entries(
    max_daily_loss_points, expected_loss
):
    class LossCapLogic(_FakeLogic):
        class EMATrendSignalEngine:
            def __init__(self, _config):
                pass

            def evaluate_candle(self, candles, position=None):
                timestamp = pd.Timestamp(candles.iloc[-1]["timestamp"])
                if position is None and timestamp in {
                    pd.Timestamp("2026-10-15T09:25:00+05:30"),
                    pd.Timestamp("2026-10-15T09:35:00+05:30"),
                }:
                    return _FakeLogic.EMATrendDecision("ENTER_LONG", signal_triggered=True)
                return _FakeLogic.EMATrendDecision()

    timestamps = pd.date_range("2026-10-15T09:15:00+05:30", periods=7, freq="5min")
    closes = [22000.0, 22000.0, 22000.0, 22000.0, 22000.0, 22000.0, 22000.0]
    opens = closes.copy()
    highs = [value + 5 for value in closes]
    lows = [value - 5 for value in closes]
    closes[3] = 21820.0
    opens[3] = 22000.0
    lows[3] = 21800.0
    index_bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [5.0] * len(timestamps),
        }
    )

    replay_options = (
        {"max_daily_loss_points": max_daily_loss_points}
        if max_daily_loss_points is not None
        else {}
    )
    trades, counters = replay_index_only(
        signal_logic=LossCapLogic,
        strategy_config=None,
        index_bars=index_bars,
        start=date(2026, 10, 15),
        end=date(2026, 10, 15),
        poll_seconds=30,
        **replay_options,
    )

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "DAILY_MAX_LOSS"
    assert trades.iloc[0]["index_points"] == expected_loss
    assert counters == {"signals": 2, "entries": 1, "exits": 1, "daily_loss_halts": 1}


def test_index_only_daily_loss_cap_must_be_positive_and_finite():
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-10-15T09:15:00+05:30", periods=1, freq="5min"),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [5.0],
        }
    )

    with pytest.raises(ValueError, match="max_daily_loss_points must be positive and finite"):
        replay_index_only(
            signal_logic=_FakeLogic,
            strategy_config=None,
            index_bars=bars,
            start=date(2026, 10, 15),
            end=date(2026, 10, 15),
            max_daily_loss_points=0,
        )


@pytest.mark.parametrize(
    ("trailing_stop", "expected_exit"),
    [
        ("swing_2bar", 980.0),
        ("atr_2x", 1080.0),
        ("breakeven_1r_ema9", 1000.0),
        ("breakeven_1r_ema11", 1060.0),
        ("combined", 1080.0),
    ],
)
def test_index_only_trailing_stops_tighten_and_exit_on_a_later_bar(trailing_stop, expected_exit):
    class SwingTrailLogic(_FakeLogic):
        class EMATrendSignalEngine:
            def __init__(self, _config):
                pass

            def evaluate_candle(self, candles, position=None):
                if (
                    position is None
                    and pd.Timestamp(candles.iloc[-1]["timestamp"])
                    == pd.Timestamp("2026-10-15T09:20:00+05:30")
                ):
                    return _FakeLogic.EMATrendDecision("ENTER_LONG", signal_triggered=True)
                return _FakeLogic.EMATrendDecision()

        @staticmethod
        def build_ema_trend_with_indicators(bars, _config):
            frame = bars.copy()
            frame["ema9"] = 900.0
            frame["ema11"] = [950.0, 950.0, 1060.0, 1070.0, 1080.0]
            frame["atr"] = 20.0
            return frame

    timestamps = pd.date_range("2026-10-15T09:15:00+05:30", periods=5, freq="5min")
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1000.0, 1000.0, 1000.0, 1100.0, 990.0],
            "high": [1010.0, 1010.0, 1120.0, 1120.0, 1000.0],
            "low": [990.0, 990.0, 980.0, 975.0, 980.0],
            "close": [1000.0, 1000.0, 1040.0, 995.0, 990.0],
            "volume": [5.0] * len(timestamps),
        }
    )

    trades, counters = replay_index_only(
        signal_logic=SwingTrailLogic,
        strategy_config=None,
        index_bars=bars,
        start=date(2026, 10, 15),
        end=date(2026, 10, 15),
        poll_seconds=30,
        trailing_stop=trailing_stop,
    )

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "TRAILING_STOP"
    assert trades.iloc[0]["exit_index"] == expected_exit
    assert trades.iloc[0]["index_points"] == expected_exit - 1000.0
    assert trades.iloc[0]["exit_time"] == "2026-10-15T09:35:00+05:30"
    assert counters["exits"] == 1


def test_index_only_breakeven_trail_ignores_entry_bar_high_before_entry():
    class EntryBarHighLogic(_FakeLogic):
        class EMATrendSignalEngine:
            def __init__(self, _config):
                pass

            def evaluate_candle(self, candles, position=None):
                if (
                    position is None
                    and pd.Timestamp(candles.iloc[-1]["timestamp"])
                    == pd.Timestamp("2026-10-15T09:20:00+05:30")
                ):
                    return _FakeLogic.EMATrendDecision("ENTER_LONG", signal_triggered=True)
                return _FakeLogic.EMATrendDecision()

        @staticmethod
        def build_ema_trend_with_indicators(bars, _config):
            frame = bars.copy()
            frame["ema9"] = [990.0, 990.0, 1004.0, 1008.0]
            frame["atr"] = 20.0
            return frame

    timestamps = pd.date_range("2026-10-15T09:15:00+05:30", periods=4, freq="5min")
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1000.0, 1000.0, 1000.0, 1002.0],
            "high": [1005.0, 1050.0, 1005.0, 1009.0],
            "low": [995.0, 995.0, 995.0, 1000.0],
            "close": [1000.0, 1000.0, 1002.0, 1005.0],
            "volume": [5.0] * len(timestamps),
        }
    )

    trades, _ = replay_index_only(
        signal_logic=EntryBarHighLogic,
        strategy_config=None,
        index_bars=bars,
        start=date(2026, 10, 15),
        end=date(2026, 10, 15),
        poll_seconds=30,
        trailing_stop="breakeven_1r_ema9",
    )

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "SQUARE_OFF"


def test_history_cache_refresh_replaces_cached_candles(tmp_path):
    class ChangingAPI(_FakeAPI):
        def history(self, symbol, **kwargs):
            self.requests.append((symbol, kwargs))
            close = 100.0 + len(self.requests)
            return pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2026-10-15T09:15:00+05:30"]),
                    "open": [close],
                    "high": [close],
                    "low": [close],
                    "close": [close],
                    "volume": [1.0],
                }
            )

    api = ChangingAPI(pd.DataFrame())
    cache_dir = tmp_path
    request = {
        "resolution": "1",
        "start": date(2026, 10, 15),
        "end": date(2026, 10, 15),
    }

    FyersHistoryCache(cache_dir).history(api, "NSE:NIFTY50-INDEX", **request)
    refreshed = FyersHistoryCache(cache_dir, refresh=True).history(
        api,
        "NSE:NIFTY50-INDEX",
        **request,
    )
    cached = FyersHistoryCache(cache_dir).history(api, "NSE:NIFTY50-INDEX", **request)

    assert len(api.requests) == 2
    assert refreshed.iloc[0]["close"] == 102.0
    assert cached.iloc[0]["close"] == 102.0


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
    assert trades.iloc[0]["exit_reason"] == "EMA9_EXIT"
    assert api.requests[0][0] == "NSE:NIFTY26O2022500CE"


def test_daily_max_loss_exits_position_and_resets_for_the_next_day():
    class DailyEntryLogic(_FakeLogic):
        class EMATrendSignalEngine:
            def __init__(self, _config):
                pass

            def evaluate_candle(self, candles, position=None):
                timestamp = pd.Timestamp(candles.iloc[-1]["timestamp"])
                if position is None and timestamp.time() in {time(9, 25), time(9, 35)}:
                    return _FakeLogic.EMATrendDecision("ENTER_LONG", signal_triggered=True)
                return _FakeLogic.EMATrendDecision()

    days = [date(2026, 10, 15), date(2026, 10, 16)]
    index_timestamps = [
        pd.Timestamp(f"{day.isoformat()}T{clock}+05:30")
        for day in days
        for clock in ("09:25:00", "09:30:00", "09:35:00")
    ]
    index_bars = pd.DataFrame(
        {
            "timestamp": index_timestamps,
            "open": [22500.0] * len(index_timestamps),
            "high": [22520.0] * len(index_timestamps),
            "low": [22490.0] * len(index_timestamps),
            "close": [22510.0] * len(index_timestamps),
            "volume": [5] * len(index_timestamps),
        }
    )
    option_timestamps = pd.DatetimeIndex(
        [
            stamp
            for day in days
            for stamp in pd.date_range(
                pd.Timestamp(f"{day.isoformat()}T09:15:00+05:30"),
                pd.Timestamp(f"{day.isoformat()}T15:15:00+05:30"),
                freq="5s",
            )
        ]
    )
    option_prices = [200.0 if stamp.time() < time(9, 30) else 115.0 for stamp in option_timestamps]
    option_bars = pd.DataFrame(
        {
            "timestamp": option_timestamps,
            "open": option_prices,
            "high": option_prices,
            "low": option_prices,
            "close": option_prices,
            "volume": 1,
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
        signal_logic=DailyEntryLogic,
        strategy_config=None,
        index_bars=index_bars,
        api=api,
        active_rows=active_rows,
        expiries=[date(2026, 10, 20)],
        start=days[0],
        end=days[1],
        max_daily_loss=5500.0,
        poll_seconds=30,
    )

    assert counters == {"signals": 4, "entries": 2, "exits": 2, "daily_loss_halts": 2}
    assert trades["exit_reason"].tolist() == ["DAILY_MAX_LOSS", "DAILY_MAX_LOSS"]
    assert trades["gross_pnl"].tolist() == [-5525.0, -5525.0]
    assert trades["entry_time"].str[:10].tolist() == ["2026-10-15", "2026-10-16"]
