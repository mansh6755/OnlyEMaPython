from datetime import UTC, date, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import talib

from Dependencies.fyers_nifty_ema9_touch_backtest import (
    FyersHistoryCache,
    _completed_option_quotes,
    _quote_index_at_or_before,
    _select_itm_contract,
    replay_ema9_touch_index,
)
from Dependencies.fyers_option_backtest import _archived_option_rows
from Dependencies.fyers_options import FyersOptionContract
from Dependencies import fyers_nifty_ema9_touch_backtest as backtest_module


class _SignalLogic:
    class EMATrendSignalEngine:
        def __init__(self, _config):
            self.calls = 0

        def evaluate_candle(self, _candles):
            self.calls += 1
            action = "ENTER_LONG" if self.calls == 9 else "HOLD"
            return SimpleNamespace(action=action)

    @staticmethod
    def build_ema_trend_with_indicators(bars, _config):
        return bars.copy()


@pytest.mark.parametrize("instrument", ["NIFTY", "SENSEX"])
def test_replay_exits_long_at_prior_five_minute_ema9_buffer_on_minute_touch(instrument):
    five_minute = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-10-19 09:15:00",
                periods=12,
                freq="5min",
                tz="Asia/Kolkata",
            ),
            "open": np.arange(100.0, 160.0, 5.0),
            "high": np.arange(101.0, 161.0, 5.0),
            "low": np.arange(99.0, 159.0, 5.0),
            "close": np.arange(100.0, 160.0, 5.0),
            "volume": 1,
        }
    )
    minute_timestamps = pd.date_range(
        "2026-10-19 09:15:00",
        periods=60,
        freq="min",
        tz="Asia/Kolkata",
    )
    minutes = pd.DataFrame(
        {
            "timestamp": minute_timestamps,
            "open": 150.0,
            "high": 151.0,
            "low": 149.0,
            "close": 150.0,
            "volume": 1,
        }
    )
    prior_ema9 = float(talib.EMA(five_minute["close"].to_numpy(), timeperiod=9)[9])
    prior_stop = prior_ema9 - 3.0
    minutes.loc[minutes["timestamp"] == pd.Timestamp("2026-10-19 10:05:00", tz="Asia/Kolkata"), "low"] = (
        prior_stop - 1.0
    )

    trades = replay_ema9_touch_index(
        signal_logic=_SignalLogic,
        strategy_config=None,
        five_minute_bars=five_minute,
        minute_bars=minutes,
        start=date(2026, 10, 19),
        end=date(2026, 10, 19),
        instrument=instrument,
    )

    assert len(trades) == 1
    assert trades[0]["signal"] == "ENTER_LONG"
    assert trades[0]["exit_reason"] == "EMA9_TOUCH"
    assert trades[0]["entry_time"] == "2026-10-19T10:05:00+05:30"
    assert trades[0]["exit_time"] == "2026-10-19T10:06:00+05:30"
    assert trades[0]["exit_index"] == prior_stop


def test_archived_option_symbols_parse_for_sensex_expiry():
    expiry = date(2026, 10, 29)

    rows = _archived_option_rows(
        [
            "BSE:SENSEX26OCT2986000CE",
            "BSE:SENSEX26OCT2986000PE",
        ],
        expiry,
        20,
        underlying="SENSEX",
    )

    assert [(row["option_type"], row["strike_price"]) for row in rows] == [
        ("CE", 86000.0),
        ("PE", 86000.0),
    ]


def test_backtest_routes_sensex_to_bse_market_data(monkeypatch, tmp_path):
    calls = {}

    class FakeCache:
        hits = 0
        misses = 0

        def __init__(self, *_args, **_kwargs):
            pass

    class FakeApi:
        def option_master(self, exchange):
            calls["exchange"] = exchange
            return [
                {
                    "underlying": "SENSEX",
                    "option_type": "CE",
                    "expiry": int(datetime(2026, 10, 29, tzinfo=UTC).timestamp()),
                    "lot_size": 20,
                }
            ]

    monkeypatch.setattr(backtest_module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(backtest_module, "FyersHistoryCache", FakeCache)
    monkeypatch.setattr(
        backtest_module,
        "_fetch_index_history_with_cache",
        lambda _api, _start, _end, _cache, **kwargs: calls.update(
            index_symbol=kwargs["index_symbol"]
        )
        or pd.DataFrame(),
    )
    monkeypatch.setattr(backtest_module, "_completed_five_minute_bars", lambda _bars: pd.DataFrame())
    monkeypatch.setattr(
        backtest_module,
        "replay_ema9_touch_index",
        lambda **kwargs: calls.update(replay_instrument=kwargs["instrument"]) or [],
    )
    monkeypatch.setattr(backtest_module, "_fetch_expiry_dates", lambda *_args, **kwargs: [])
    monkeypatch.setattr(
        backtest_module,
        "attach_option_prices",
        lambda _trades, **kwargs: (
            calls.update(
                active_underlying=kwargs["underlying"],
                history_symbol=kwargs["index_symbol"],
            )
            or pd.DataFrame(),
            {"missing_option_data": 0, "daily_loss_halts": 0},
        ),
    )

    trades, _counters, index_trades = backtest_module.run_backtest(
        FakeApi(),
        signal_logic=None,
        strategy_config=None,
        start=date(2026, 10, 19),
        end=date(2026, 10, 23),
        instrument="SENSEX",
    )

    assert trades.empty
    assert index_trades == 0
    assert calls == {
        "index_symbol": "BSE:SENSEX-INDEX",
        "replay_instrument": "SENSEX",
        "exchange": "BSE",
        "active_underlying": "SENSEX",
        "history_symbol": "BSE:SENSEX-INDEX",
    }


def test_select_itm_contract_uses_nearest_upcoming_expiry_and_two_itm_steps():
    rows = [
        {
            "underlying": "NIFTY",
            "symbol": f"NSE:NIFTY26OCT27{strike}CE",
            "expiry": int(datetime(2026, 10, 27, tzinfo=UTC).timestamp()),
            "strike_price": strike,
            "option_type": "CE",
            "lot_size": 65,
        }
        for strike in (22500, 22550, 22600, 22650, 22700)
    ]

    contract = _select_itm_contract(
        None,
        action="ENTER_LONG",
        spot=22605,
        trade_day=date(2026, 10, 26),
        active_rows=rows,
        expiries=[date(2026, 10, 27), date(2026, 11, 3)],
        archived_symbols={},
        fallback_lot_size=65,
        strike_steps=2,
    )

    assert contract == FyersOptionContract(
        "NSE:NIFTY26OCT2722500CE",
        date(2026, 10, 27),
        "CE",
        22500,
        65,
    )


def test_completed_option_quote_lookup_uses_timezone_aware_datetime_units():
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01T09:15:00+05:30",
                    "2025-01-01T09:15:05+05:30",
                    "2025-01-01T09:15:10+05:30",
                ]
            ),
            "open": [100.0, 101.0, 102.0],
            "high": [100.0, 101.0, 102.0],
            "low": [100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0],
        }
    )
    quotes = _completed_option_quotes(bars)

    index = _quote_index_at_or_before(
        quotes,
        pd.Timestamp("2025-01-01T09:15:10+05:30"),
    )

    assert index == 1
    assert quotes.iloc[index]["close"] == 101.0


def test_fyers_history_cache_reuses_persisted_candles(tmp_path):
    class _API:
        def __init__(self):
            self.calls = 0

        def history(self, _symbol, **_kwargs):
            self.calls += 1
            return pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2026-10-23T09:15:00+05:30"]),
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [10.0],
                }
            )

    api = _API()
    cache = FyersHistoryCache(tmp_path)
    kwargs = {
        "resolution": "5S",
        "start": date(2026, 10, 23),
        "end": date(2026, 10, 23),
        "expired": True,
    }

    first = cache.history(api, "NSE:NIFTY26OCT2722500CE", **kwargs)
    second = cache.history(api, "NSE:NIFTY26OCT2722500CE", **kwargs)

    assert api.calls == 1
    assert cache.hits == 1
    assert cache.misses == 1
    pd.testing.assert_frame_equal(first, second)
