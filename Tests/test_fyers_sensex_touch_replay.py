import pandas as pd
import pytest

from Dependencies.fyers_sensex_touch_replay import (
    _archived_sensex_option_rows,
    _attach_option_prices,
    _select_option_strike,
    replay_sensex_touch_cap,
)


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
            "open": [1000.0] * 10,
            "high": [1001.0] * 10,
            "low": [98.5] * 5 + [97.5] * 5,
            "close": [1000.0] * 10,
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


def test_daily_cap_exits_at_minute_close_before_five_minute_close():
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
            "open": [1000.0] * 3,
            "high": [1000.0] * 3,
            "low": [1000.0, 999.0, 695.0],
            "close": [1000.0, 1000.0, 700.0],
            "ema9": [500.0] * 3,
        }
    )
    minute_times = pd.date_range("2026-01-01T09:25:00+05:30", periods=5, freq="min")
    minute_bars = pd.DataFrame(
        {
            "timestamp": minute_times,
            "open": [800.0, 750.0, 700.0, 700.0, 700.0],
            "high": [805.0] * 5,
            "low": [795.0, 695.0, 695.0, 695.0, 695.0],
            "close": [800.0, 700.0, 700.0, 700.0, 700.0],
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
    assert trades.iloc[0]["exit_reason"] == "DAILY_MAX_LOSS"
    assert trades.iloc[0]["exit_time"] == "2026-01-01T09:27:00+05:30"
    assert trades.iloc[0]["exit_index"] == 700.0


def test_archived_sensex_option_symbols_resolve_weekly_expiry_and_strike():
    expiry = pd.Timestamp("2026-10-08").date()
    symbols = [
        "BSE:SENSEX26O0869600CE",
        "BSE:SENSEX26O0869600PE",
        "BSE:SENSEX26O1569600CE",
    ]

    rows = _archived_sensex_option_rows(symbols, expiry, lot_size=20)

    assert rows == [
        {
            "symbol": "BSE:SENSEX26O0869600CE",
            "option_type": "CE",
            "strike_price": 69600.0,
            "lot_size": 20,
            "expiry": expiry,
            "symbol_prefix": "BSE:SENSEX26O08",
        },
        {
            "symbol": "BSE:SENSEX26O0869600PE",
            "option_type": "PE",
            "strike_price": 69600.0,
            "lot_size": 20,
            "expiry": expiry,
            "symbol_prefix": "BSE:SENSEX26O08",
        },
    ]


def test_option_price_capture_uses_completed_candles_without_lookahead():
    from datetime import UTC, date, datetime, timedelta

    from Dependencies.fyers_ema_replay import IST

    trade_day = date.today() - timedelta(days=60)
    expiry = trade_day + timedelta(days=2)
    active_expiry = date.today() + timedelta(days=60)
    active_expiry_timestamp = int(
        datetime.combine(active_expiry, datetime.min.time(), tzinfo=UTC).timestamp()
    )
    entry_time = pd.Timestamp(datetime.combine(trade_day, datetime.min.time()).replace(hour=9, minute=25), tz=IST)
    exit_time = entry_time + pd.Timedelta(minutes=1)

    class FakeApi:
        def option_master(self, _exchange):
            return [
                {
                    "underlying": "SENSEX",
                    "option_type": "CE",
                    "expiry": active_expiry_timestamp,
                    "lot_size": 20,
                    "symbol": "BSE:SENSEX26OCT69600CE",
                    "strike_price": 69600.0,
                }
            ]

        def expired_option_expiries(self, _symbol, _start, _end):
            return [expiry]

        def expired_option_symbols(self, _symbol, requested_expiry):
            assert requested_expiry == expiry
            prefix = f"SENSEX{expiry:%y}{expiry:%b}".upper()
            return [f"BSE:{prefix}69600CE"]

    class FakeCache:
        def history(self, _api, symbol, *, resolution, start, end, expired):
            assert symbol == f"BSE:SENSEX{expiry:%y}{expiry:%b}".upper() + "69600CE"
            assert resolution == "5S"
            assert start == end == trade_day
            assert expired is True
            return pd.DataFrame(
                {
                    "timestamp": [
                        entry_time - pd.Timedelta(seconds=10),
                        entry_time - pd.Timedelta(seconds=5),
                        entry_time,
                        exit_time - pd.Timedelta(seconds=5),
                    ],
                    "close": [90.0, 101.0, 999.0, 120.0],
                }
            )

    trades = pd.DataFrame(
        [
            {
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "signal": "ENTER_LONG",
                "entry_index": 69601.0,
            }
        ]
    )
    marked = _attach_option_prices(
        trades,
        api=FakeApi(),
        cache=FakeCache(),
        start=trade_day,
        end=trade_day,
        resolution="5S",
    )

    assert marked.loc[0, "option_type"] == "CE"
    assert marked.loc[0, "entry_option_price"] == 101.0
    assert marked.loc[0, "entry_option_candle_time"] == entry_time.isoformat()
    assert marked.loc[0, "exit_option_price"] == 120.0
    assert marked.loc[0, "estimated_option_pnl"] == 380.0
    assert marked.loc[0, "option_price_status"] == "OK"

    class MissingCache:
        def history(self, *_args, **_kwargs):
            return pd.DataFrame({"timestamp": [exit_time + pd.Timedelta(seconds=5)], "close": [130.0]})

    missing = _attach_option_prices(
        trades,
        api=FakeApi(),
        cache=MissingCache(),
        start=trade_day,
        end=trade_day,
        resolution="5S",
    )

    assert missing.loc[0, "option_price_status"] == "MISSING_ENTRY_AND_EXIT_CANDLE"
    assert pd.isna(missing.loc[0, "entry_option_price"])
    assert pd.isna(missing.loc[0, "exit_option_price"])


def test_option_daily_loss_cap_exits_on_five_second_mark_and_halts_day():
    from datetime import UTC, date, datetime, timedelta

    from Dependencies.fyers_ema_replay import IST

    trade_day = date.today() - timedelta(days=60)
    expiry = trade_day + timedelta(days=2)
    active_expiry = date.today() + timedelta(days=60)
    active_expiry_timestamp = int(
        datetime.combine(active_expiry, datetime.min.time(), tzinfo=UTC).timestamp()
    )
    entry_time = pd.Timestamp(
        datetime.combine(trade_day, datetime.min.time()).replace(hour=9, minute=25),
        tz=IST,
    )
    planned_exit = entry_time + pd.Timedelta(minutes=1)

    class FakeApi:
        def option_master(self, _exchange):
            return [
                {
                    "underlying": "SENSEX",
                    "option_type": "CE",
                    "expiry": active_expiry_timestamp,
                    "lot_size": 20,
                    "symbol": "BSE:SENSEX26OCT69600CE",
                    "strike_price": 69600.0,
                }
            ]

        def expired_option_expiries(self, _symbol, _start, _end):
            return [expiry]

        def expired_option_symbols(self, _symbol, requested_expiry):
            assert requested_expiry == expiry
            return [f"BSE:SENSEX{expiry:%y}{expiry:%b}".upper() + "69600CE"]

    class FakeCache:
        def history(self, _api, _symbol, *, resolution, start, end, expired):
            assert resolution == "5S"
            assert start == end == trade_day
            assert expired is True
            return pd.DataFrame(
                {
                    "timestamp": [
                        entry_time - pd.Timedelta(seconds=5),
                        entry_time,
                        entry_time + pd.Timedelta(seconds=5),
                    ],
                    "close": [101.0, 999.0, 40.0],
                }
            )

    trades = pd.DataFrame(
        [
            {
                "entry_time": entry_time.isoformat(),
                "exit_time": planned_exit.isoformat(),
                "signal": "ENTER_LONG",
                "entry_index": 69601.0,
                "exit_index": 69650.0,
                "exit_reason": "EMA9_TOUCH",
                "index_points": 49.0,
            },
            {
                "entry_time": (planned_exit + pd.Timedelta(minutes=1)).isoformat(),
                "exit_time": (planned_exit + pd.Timedelta(minutes=2)).isoformat(),
                "signal": "ENTER_LONG",
                "entry_index": 69600.0,
                "exit_index": 69610.0,
                "exit_reason": "EMA9_TOUCH",
                "index_points": 10.0,
            },
        ]
    )
    minute_bars = pd.DataFrame(
        {
            "timestamp": [entry_time - pd.Timedelta(minutes=1)],
            "close": [69601.0],
        }
    )

    marked = _attach_option_prices(
        trades,
        api=FakeApi(),
        cache=FakeCache(),
        start=trade_day,
        end=trade_day,
        resolution="5S",
        option_daily_loss_cap=1000.0,
        minute_bars=minute_bars,
    )

    assert len(marked) == 1
    assert marked.loc[0, "exit_reason"] == "OPTION_DAILY_MAX_LOSS"
    assert marked.loc[0, "exit_time"] == (entry_time + pd.Timedelta(seconds=10)).isoformat()
    assert marked.loc[0, "exit_option_price"] == 40.0
    assert marked.loc[0, "estimated_option_pnl"] == -1220.0
    assert marked.loc[0, "index_exit_mark_basis"] == "LAST_COMPLETED_1M_CLOSE"


def test_option_strike_steps_select_directional_itm_and_otm_contracts():
    from datetime import date

    expiry = date(2026, 10, 29)
    options = [
        {
            "symbol": f"BSE:SENSEX{strike}{option_type}",
            "option_type": option_type,
            "strike_price": strike,
            "lot_size": 20,
            "expiry": expiry,
        }
        for option_type in ("CE", "PE")
        for strike in (69400, 69500, 69600, 69700, 69800)
    ]

    assert _select_option_strike(
        options,
        action="ENTER_LONG",
        spot=69649,
        expiry=expiry,
        moneyness_steps=0,
    ).strike == 69600
    assert _select_option_strike(
        options,
        action="ENTER_LONG",
        spot=69649,
        expiry=expiry,
        moneyness_steps=2,
    ).strike == 69400
    assert _select_option_strike(
        options,
        action="ENTER_LONG",
        spot=69649,
        expiry=expiry,
        moneyness_steps=-2,
    ).strike == 69800
    assert _select_option_strike(
        options,
        action="ENTER_SHORT",
        spot=69649,
        expiry=expiry,
        moneyness_steps=2,
    ).strike == 69800
    assert _select_option_strike(
        options,
        action="ENTER_SHORT",
        spot=69649,
        expiry=expiry,
        moneyness_steps=-2,
    ).strike == 69400


def test_option_strike_steps_reject_unlisted_ladder_step():
    from datetime import date

    expiry = date(2026, 10, 29)
    options = [
        {
            "symbol": f"BSE:SENSEX{strike}CE",
            "option_type": "CE",
            "strike_price": strike,
            "lot_size": 20,
        }
        for strike in (69500, 69600, 69700)
    ]

    with pytest.raises(ValueError, match="No 2-step ITM CE contract"):
        _select_option_strike(
            options,
            action="ENTER_LONG",
            spot=69600,
            expiry=expiry,
            moneyness_steps=2,
        )


def test_archived_sparse_symbol_list_reconstructs_atm_and_adjacent_strikes():
    from datetime import date

    expiry = date(2026, 9, 24)
    sparse_rows = _archived_sensex_option_rows(
        [
            "BSE:SENSEX26SEP75000CE",
            "BSE:SENSEX26SEP75000PE",
            "BSE:SENSEX26SEP76000CE",
            "BSE:SENSEX26SEP76000PE",
        ],
        expiry,
        lot_size=20,
    )

    assert _select_option_strike(
        sparse_rows,
        action="ENTER_LONG",
        spot=74450,
        expiry=expiry,
        moneyness_steps=0,
        strike_step_points=100,
    ).symbol == "BSE:SENSEX26SEP74400CE"
    assert _select_option_strike(
        sparse_rows,
        action="ENTER_LONG",
        spot=74450,
        expiry=expiry,
        moneyness_steps=2,
        strike_step_points=100,
    ).symbol == "BSE:SENSEX26SEP74200CE"
    assert _select_option_strike(
        sparse_rows,
        action="ENTER_SHORT",
        spot=74450,
        expiry=expiry,
        moneyness_steps=2,
        strike_step_points=100,
    ).symbol == "BSE:SENSEX26SEP74600PE"
