import argparse
from datetime import date, time

import pandas as pd
import pytest

from Dependencies.fyers_ema_replay import (
    FyersHistoryCache,
    _action_polls,
    _archived_option_rows,
    _expiry_lookup_end,
    _parse_boolean_flag,
    _parse_cache_flag,
    _quote_at,
    _report,
    _replay_poll_seconds,
    _resolve_contract,
    replay,
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


def test_expiry_lookup_includes_the_following_week_after_the_replay_end():
    assert _expiry_lookup_end(date(2026, 2, 28), date(2026, 10, 15)) == date(2026, 3, 14)
    assert _expiry_lookup_end(date(2026, 10, 15), date(2026, 10, 15)) == date(2026, 10, 14)


def test_archived_sensex_symbols_are_resolved_with_the_bse_prefix_and_lot_size():
    rows = _archived_option_rows(
        ["BSE:SENSEX2612285800CE", "BSE:SENSEX2612285800PE"],
        date(2026, 1, 22),
        20,
        underlying="SENSEX",
    )

    assert rows == [
        {
            "symbol": "BSE:SENSEX2612285800CE",
            "option_type": "CE",
            "strike_price": 85800.0,
            "lot_size": 20,
            "expiry": date(2026, 1, 22),
        },
        {
            "symbol": "BSE:SENSEX2612285800PE",
            "option_type": "PE",
            "strike_price": 85800.0,
            "lot_size": 20,
            "expiry": date(2026, 1, 22),
        },
    ]


def test_use_cache_flag_accepts_true_or_false_values():
    assert _parse_cache_flag("true") is True
    assert _parse_cache_flag("FALSE") is False
    with pytest.raises(argparse.ArgumentTypeError, match="useCache must be true or false"):
        _parse_cache_flag("yes")


def test_parent_logic_flag_accepts_true_or_false_values():
    assert _parse_boolean_flag("true") is True
    assert _parse_boolean_flag("FALSE") is False
    with pytest.raises(argparse.ArgumentTypeError, match="use true or false"):
        _parse_boolean_flag("yes")


@pytest.mark.parametrize(
    ("parent_logic", "override", "expected"),
    [(True, None, 2), (False, None, 30), (True, 15, 15), (False, 15, 15)],
)
def test_replay_poll_interval_uses_mode_default_unless_explicitly_overridden(parent_logic, override, expected):
    assert _replay_poll_seconds(parent_logic, override) == expected


def test_legacy_contract_selection_uses_next_week_and_unrounded_spot():
    expiries = [date(2027, 1, 19), date(2027, 1, 12)]
    active_rows = [
        {
            "underlying": "NIFTY",
            "symbol": f"NSE:NIFTY-{expiry.isoformat()}-{strike}-CE",
            "expiry": expiry,
            "strike_price": strike,
            "option_type": "CE",
            "lot_size": 65,
        }
        for expiry in expiries
        for strike in (82500, 82600)
    ]

    contract = _resolve_contract(
        object(),
        action="ENTER_LONG",
        spot=82551,
        trade_date=date(2027, 1, 4),
        active_rows=active_rows,
        expiries=expiries,
        archived_symbols={},
        fallback_lot_size=65,
        expiry_policy="next_week",
        strike_step=None,
    )

    assert contract.expiry == date(2027, 1, 12)
    assert contract.strike == 82600


def test_report_records_parent_logic_and_keeps_modes_in_separate_files(tmp_path, monkeypatch):
    monkeypatch.setattr("Dependencies.fyers_ema_replay.OUTPUT_DIR", tmp_path)
    counters = {"signals": 0, "entries": 0, "exits": 0, "daily_loss_halts": 0}
    parent_paths = _report(
        pd.DataFrame(),
        counters,
        date(2026, 1, 1),
        date(2026, 1, 31),
        "1",
        2,
        FyersHistoryCache(tmp_path / "cache"),
        "SENSEX",
        "nearest",
        100,
        True,
    )
    legacy_paths = _report(
        pd.DataFrame(),
        counters,
        date(2026, 1, 1),
        date(2026, 1, 31),
        "1",
        30,
        FyersHistoryCache(tmp_path / "cache"),
        "SENSEX",
        "next_week",
        None,
        False,
    )

    assert parent_paths[0].name == "fyers_ema_replay_sensex_20260101_20260131_trades.csv"
    assert legacy_paths[0].name == "fyers_ema_replay_sensex_no_parent_20260101_20260131_trades.csv"
    assert "Parent logic: true." in parent_paths[1].read_text()
    assert "Parent logic: false." in legacy_paths[1].read_text()
    assert "unrounded spot" in legacy_paths[1].read_text()


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
            "symbol": "NSE:NIFTY26O2722500CE",
            "expiry": date(2026, 10, 27),
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
        expiries=[date(2026, 10, 20), date(2026, 10, 27)],
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
    assert api.requests[0][0] == "NSE:NIFTY26O2722500CE"


def test_replay_can_use_sensex_option_contracts_without_changing_nifty_defaults():
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
            "open": [82500.0, 82510.0],
            "high": [82520.0, 82520.0],
            "low": [82490.0, 82500.0],
            "close": [82510.0, 82505.0],
            "volume": [5, 5],
        }
    )
    api = _FakeAPI(option_bars)
    active_rows = [
        {
            "underlying": "SENSEX",
            "symbol": "BSE:SENSEX26O2082500CE",
            "expiry": date(2026, 10, 20),
            "strike_price": 82500.0,
            "option_type": "CE",
            "lot_size": 20,
        },
        {
            "underlying": "SENSEX",
            "symbol": "BSE:SENSEX26O2782500CE",
            "expiry": date(2026, 10, 27),
            "strike_price": 82500.0,
            "option_type": "CE",
            "lot_size": 20,
        },
    ]

    trades, counters = replay(
        signal_logic=_FakeLogic,
        strategy_config=None,
        index_bars=index_bars,
        api=api,
        active_rows=active_rows,
        expiries=[date(2026, 10, 20), date(2026, 10, 27)],
        start=day,
        end=day,
        underlying="SENSEX",
        underlying_symbol="BSE:SENSEX-INDEX",
        strike_step=100,
        expiry_policy="nearest",
    )

    assert counters == {"signals": 1, "entries": 1, "exits": 1, "daily_loss_halts": 0}
    assert trades.iloc[0]["symbol"] == "BSE:SENSEX26O2082500CE"
    assert trades.iloc[0]["quantity"] == 20
    assert trades.iloc[0]["gross_pnl"] == 200.0
    assert api.requests[0][0] == "BSE:SENSEX26O2082500CE"


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
            "symbol": "NSE:NIFTY26O2722500CE",
            "expiry": date(2026, 10, 27),
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
        expiries=[date(2026, 10, 20), date(2026, 10, 27)],
        start=days[0],
        end=days[1],
        max_daily_loss=5500.0,
        poll_seconds=30,
    )

    assert counters == {"signals": 4, "entries": 2, "exits": 2, "daily_loss_halts": 2}
    assert trades["exit_reason"].tolist() == ["DAILY_MAX_LOSS", "DAILY_MAX_LOSS"]
    assert trades["gross_pnl"].tolist() == [-5525.0, -5525.0]
    assert trades["entry_time"].str[:10].tolist() == ["2026-10-15", "2026-10-16"]
