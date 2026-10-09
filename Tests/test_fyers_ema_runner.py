from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from Dependencies.broker_contract import normalize_order_result
from Dependencies.fyers_ema_runner import (
    ActivePosition,
    _close_position,
    _entry_bar_is_fresh,
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
    )

    pnl = _close_position(
        FakeAPI(),
        FakeExecution(),
        position,
        live_enabled=True,
        product_type="INTRADAY",
        reason="SQUARE_OFF",
    )

    assert pnl == -650
