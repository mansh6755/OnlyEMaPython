from datetime import date

import pytest

from Dependencies.fyers_options import (
    select_atm_option,
    select_next_week_atm_option,
    select_next_week_expiry,
    select_parent_style_atm_option,
    select_parent_style_expiry,
)


def test_select_next_week_expiry_skips_current_week_and_sorts_candidates():
    selected = select_next_week_expiry(
        [
            {"date": "2026-10-08"},
            {"date": "2026-10-15"},
            {"date": "2026-10-13"},
            {"date": "2026-10-20"},
        ],
        today=date(2026, 10, 9),
    )

    assert selected == date(2026, 10, 13)


def test_select_next_week_expiry_fails_when_fyers_has_none():
    with pytest.raises(ValueError, match="no listed expiry"):
        select_next_week_expiry(["2026-10-08", "2026-10-20"], today=date(2026, 10, 9))


def test_parent_style_expiry_selects_second_listed_expiry_on_or_after_today():
    selected = select_parent_style_expiry(
        ["2026-10-08", "2026-10-20", "2026-10-13", "2026-10-27"],
        today=date(2026, 10, 9),
    )

    assert selected == date(2026, 10, 20)


def test_parent_style_expiry_requires_two_future_listed_expiries():
    with pytest.raises(ValueError, match="fewer than two listed expiries"):
        select_parent_style_expiry(["2026-10-20"], today=date(2026, 10, 9))


@pytest.mark.parametrize(
    ("action", "expected_type"),
    [("ENTER_LONG", "CE"), ("ENTER_SHORT", "PE")],
)
def test_select_atm_option_maps_direction_and_uses_one_contract_lot(action, expected_type):
    selected = select_atm_option(
        [
            {"symbol": "NSE:NIFTY-X-24000-CE", "option_type": "CE", "strike_price": 24000, "lot_size": 65},
            {"symbol": "NSE:NIFTY-X-24100-CE", "option_type": "CE", "strike_price": 24100, "lot_size": 65},
            {"symbol": "NSE:NIFTY-X-24000-PE", "option_type": "PE", "strike_price": 24000, "lot_size": 65},
            {"symbol": "NSE:NIFTY-X-24100-PE", "option_type": "PE", "strike_price": 24100, "lot_size": 65},
        ],
        action=action,
        spot=24050,
        expiry=date(2026, 10, 13),
    )

    assert selected.option_type == expected_type
    assert selected.strike == 24000
    assert selected.lot_size == 65
    assert selected.expiry == date(2026, 10, 13)


def test_select_atm_option_rejects_missing_lot_size_instead_of_guessing():
    with pytest.raises(ValueError, match="no valid CE contracts"):
        select_atm_option(
            [{"symbol": "NSE:NIFTY-X-24000-CE", "option_type": "CE", "strike_price": 24000}],
            action="ENTER_LONG",
            spot=24000,
            expiry=date(2026, 10, 13),
        )


def test_select_next_week_atm_option_ignores_other_expiries():
    selected = select_next_week_atm_option(
        [
            {
                "symbol": "NSE:NIFTY26O0824000CE",
                "expiry": 1791450600,
                "option_type": "CE",
                "strike_price": 24000,
                "lot_size": 65,
            },
            {
                "symbol": "NSE:NIFTY26O1324100CE",
                "expiry": 1791886200,
                "option_type": "CE",
                "strike_price": 24100,
                "lot_size": 65,
            },
            {
                "symbol": "NSE:NIFTY26O1324000PE",
                "expiry": 1791886200,
                "option_type": "PE",
                "strike_price": 24000,
                "lot_size": 65,
            },
        ],
        action="ENTER_LONG",
        spot=24090,
        today=date(2026, 10, 9),
    )

    assert selected.symbol == "NSE:NIFTY26O1324100CE"
    assert selected.expiry == date(2026, 10, 13)


def test_parent_style_atm_rounds_spot_then_uses_lower_strike_for_tie():
    selected = select_parent_style_atm_option(
        [
            {
                "symbol": "NSE:NIFTY26O1322450CE",
                "expiry": "2026-10-13",
                "option_type": "CE",
                "strike_price": 22450,
                "lot_size": 65,
            },
            {
                "symbol": "NSE:NIFTY26O2022450CE",
                "expiry": "2026-10-20",
                "option_type": "CE",
                "strike_price": 22450,
                "lot_size": 65,
            },
            {
                "symbol": "NSE:NIFTY26O2022550CE",
                "expiry": "2026-10-20",
                "option_type": "CE",
                "strike_price": 22550,
                "lot_size": 65,
            },
            {
                "symbol": "NSE:NIFTY26O2722450CE",
                "expiry": "2026-10-27",
                "option_type": "CE",
                "strike_price": 22450,
                "lot_size": 65,
            },
        ],
        action="ENTER_LONG",
        spot=22524,
        today=date(2026, 10, 9),
    )

    assert selected.symbol == "NSE:NIFTY26O2022450CE"
    assert selected.expiry == date(2026, 10, 20)
