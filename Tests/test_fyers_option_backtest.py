from datetime import date

import pandas as pd

from Dependencies.fyers_option_backtest import (
    _archived_option_rows,
    _date_chunks,
    _expired_data_query_end,
    _fetch_index_history,
    _first_fill,
    _resolve_contract,
    replay_index_signals,
)
from Dependencies.fyers_options import FyersOptionContract


class _FakeAPI:
    def __init__(self, bars):
        self.bars = bars
        self.requests = []

    def history(self, symbol, **kwargs):
        self.requests.append((symbol, kwargs))
        return self.bars.copy()


class _FakeLogic:
    class EMATrendPositionContext:
        def __init__(self, direction, entry_underlying, stop_underlying):
            self.direction = direction
            self.entry_underlying = entry_underlying
            self.stop_underlying = stop_underlying

    class EMATrendSignalEngine:
        def __init__(self, _config):
            self.calls = 0

        def evaluate_candle(self, candles, position=None):
            from types import SimpleNamespace

            self.calls = len(candles)
            if position is None and self.calls == 2:
                return SimpleNamespace(action="ENTER_LONG", signal_triggered=True, stop_underlying=23950)
            if position is not None and self.calls == 3:
                return SimpleNamespace(action="EXIT", exit_reason="EMA9_EXIT", signal_triggered=False)
            return SimpleNamespace(action="HOLD", signal_triggered=False, exit_reason="")

    @staticmethod
    def build_ema_trend_with_indicators(bars, _config):
        return bars.copy()


def test_archived_symbols_become_selectable_contract_rows():
    rows = _archived_option_rows(
        ["NSE:NIFTY26O0622500PE", "NSE:NIFTY26O0622500CE"],
        date(2026, 10, 6),
        65,
    )

    assert len(rows) == 2
    assert {row["option_type"] for row in rows} == {"CE", "PE"}
    assert {row["strike_price"] for row in rows} == {22500.0}


def test_long_history_date_range_is_split_into_gapless_fyers_chunks():
    start = date(2025, 12, 25)
    end = date(2026, 9, 30)

    chunks = _date_chunks(start, end, chunk_days=90)

    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    assert all((chunk_end - chunk_start).days < 90 for chunk_start, chunk_end in chunks)
    assert all(chunks[index][1] + pd.Timedelta(days=1) == chunks[index + 1][0] for index in range(len(chunks) - 1))


def test_expired_data_query_stops_before_today():
    today = date(2026, 10, 10)

    assert _expired_data_query_end(date(2026, 10, 9), today) == date(2026, 10, 9)
    assert _expired_data_query_end(date(2026, 9, 1), today) == date(2026, 9, 22)


def test_index_history_chunks_are_sorted_and_deduplicated():
    class ChunkAPI:
        def __init__(self):
            self.calls = []

        def history(self, symbol, *, resolution, start, end):
            self.calls.append((symbol, resolution, start, end))
            return pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-04-01T09:15:00+05:30", "2026-04-01T09:15:00+05:30"]
                    ),
                    "open": [1, 2],
                    "high": [1, 2],
                    "low": [1, 2],
                    "close": [1, 2],
                    "volume": [1, 2],
                }
            )

    api = ChunkAPI()
    result = _fetch_index_history(api, date(2026, 1, 1), date(2026, 4, 1))

    assert len(api.calls) == 2
    assert all(call[1] == "1" for call in api.calls)
    assert result["timestamp"].is_monotonic_increasing
    assert len(result) == 1
    assert not result["timestamp"].duplicated().any()


def test_may_archived_symbol_uses_numeric_expiry_code():
    rows = _archived_option_rows(
        ["NSE:NIFTY2650517100CE", "NSE:NIFTY2650517100PE"],
        date(2026, 5, 5),
        65,
    )

    assert {row["option_type"] for row in rows} == {"CE", "PE"}
    assert {row["strike_price"] for row in rows} == {17100.0}


def test_may_monthly_archived_symbol_uses_month_name_prefix():
    rows = _archived_option_rows(
        ["NSE:NIFTY26MAY17100CE", "NSE:NIFTY26MAY17100PE"],
        date(2026, 5, 26),
        65,
    )

    assert {row["option_type"] for row in rows} == {"CE", "PE"}
    assert {row["strike_price"] for row in rows} == {17100.0}


def test_expiry_mapping_uses_second_listed_expiry_after_each_entry_date():
    archived = _archived_option_rows(
        ["NSE:NIFTY26O0622500CE"], date(2026, 10, 6), 65
    )
    active = [
        {
            "underlying": "NIFTY",
            "symbol": "NSE:NIFTY26O1322500CE",
            "expiry": 1791886200,
            "strike_price": 22500,
            "option_type": "CE",
            "lot_size": 65,
        }
    ]
    archived_symbols = {date(2026, 10, 6): [row["symbol"] for row in archived]}

    oct_first = _resolve_contract(
        None,
        action="ENTER_LONG",
        spot=22510,
        trade_date=date(2026, 10, 1),
        active_rows=active,
        expiries=[date(2026, 10, 6), date(2026, 10, 13)],
        archived_symbols=archived_symbols,
        fallback_lot_size=65,
    )
    oct_fifth = _resolve_contract(
        None,
        action="ENTER_SHORT",
        spot=22510,
        trade_date=date(2026, 10, 5),
        active_rows=[
            *active,
            {
                "underlying": "NIFTY",
                "symbol": "NSE:NIFTY26O1322500PE",
                "expiry": 1791886200,
                "strike_price": 22500,
                "option_type": "PE",
                "lot_size": 65,
            },
        ],
        expiries=[date(2026, 10, 6), date(2026, 10, 13)],
        archived_symbols=archived_symbols,
        fallback_lot_size=65,
    )

    assert oct_first == FyersOptionContract("NSE:NIFTY26O1322500CE", date(2026, 10, 13), "CE", 22500, 65)
    assert oct_fifth == FyersOptionContract("NSE:NIFTY26O1322500PE", date(2026, 10, 13), "PE", 22500, 65)


def test_first_fill_uses_option_candle_at_or_after_signal_time():
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-10-05T09:25:00+05:30", "2026-10-05T09:25:05+05:30"]
            ),
            "open": [100, 105],
            "high": [104, 108],
            "low": [99, 103],
            "close": [102, 107],
        }
    )

    fill = _first_fill(bars, pd.Timestamp("2026-10-05T09:25:05+05:30"))

    assert fill is not None
    assert fill[0] == 1
    assert fill[1]["open"] == 105


def test_replay_uses_index_signals_and_five_second_option_fill_prices():
    index_bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-10-05T09:15:00+05:30",
                    "2026-10-05T09:20:00+05:30",
                    "2026-10-05T09:25:00+05:30",
                    "2026-10-05T09:30:00+05:30",
                ]
            ),
            "open": [22500, 22510, 22520, 22515],
            "high": [22510, 22530, 22525, 22520],
            "low": [22490, 22500, 22500, 22500],
            "close": [22505, 22520, 22510, 22505],
            "volume": [1, 1, 1, 1],
        }
    )
    option_bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-10-05T09:25:00+05:30",
                    "2026-10-05T09:25:05+05:30",
                    "2026-10-05T09:30:00+05:30",
                ]
            ),
            "open": [100, 102, 115],
            "high": [104, 110, 118],
            "low": [99, 101, 113],
            "close": [102, 108, 116],
            "volume": [10, 20, 15],
        }
    )
    api = _FakeAPI(option_bars)
    active_rows = [
        {
            "underlying": "NIFTY",
            "symbol": "NSE:NIFTY26O1322500CE",
            "expiry": 1791886200,
            "strike_price": 22500,
            "option_type": "CE",
            "lot_size": 65,
        }
    ]

    trades, counters = replay_index_signals(
        signal_logic=_FakeLogic,
        strategy_config=None,
        index_bars=index_bars,
        api=api,
        active_rows=active_rows,
        expiries=[date(2026, 10, 6), date(2026, 10, 13), date(2026, 10, 19)],
        archived_symbols={date(2026, 10, 6): ["NSE:NIFTY26O0622500CE"]},
        lots=1,
        max_daily_loss=5500,
    )

    assert counters["entries"] == 1
    assert counters["exits"] == 1
    assert len(trades) == 1
    assert trades[0]["entry_option"] == 100
    assert trades[0]["exit_option"] == 115
    assert trades[0]["gross_pnl"] == 975
    assert trades[0]["symbol"] == "NSE:NIFTY26O1322500CE"


def test_replay_requests_one_minute_option_bars_when_selected():
    index_bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-10-05T09:15:00+05:30",
                    "2026-10-05T09:20:00+05:30",
                    "2026-10-05T09:25:00+05:30",
                    "2026-10-05T09:30:00+05:30",
                ]
            ),
            "open": [22500, 22510, 22520, 22515],
            "high": [22510, 22530, 22525, 22520],
            "low": [22490, 22500, 22500, 22500],
            "close": [22505, 22520, 22510, 22505],
            "volume": [1, 1, 1, 1],
        }
    )
    option_bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-10-05T09:25:00+05:30",
                    "2026-10-05T09:26:00+05:30",
                    "2026-10-05T09:30:00+05:30",
                ]
            ),
            "open": [100, 102, 115],
            "high": [104, 110, 118],
            "low": [99, 101, 113],
            "close": [102, 108, 116],
            "volume": [10, 20, 15],
        }
    )
    api = _FakeAPI(option_bars)
    active_rows = [
        {
            "underlying": "NIFTY",
            "symbol": "NSE:NIFTY26O1322500CE",
            "expiry": 1791886200,
            "strike_price": 22500,
            "option_type": "CE",
            "lot_size": 65,
        }
    ]

    trades, counters = replay_index_signals(
        signal_logic=_FakeLogic,
        strategy_config=None,
        index_bars=index_bars,
        api=api,
        active_rows=active_rows,
        expiries=[date(2026, 10, 6), date(2026, 10, 13), date(2026, 10, 19)],
        archived_symbols={date(2026, 10, 6): ["NSE:NIFTY26O0622500CE"]},
        lots=1,
        max_daily_loss=5500,
        option_resolution="1",
    )

    assert counters["entries"] == 1
    assert trades[0]["gross_pnl"] == 975
    assert trades[0]["option_resolution"] == "1"
    assert api.requests[0][1]["resolution"] == "1"
