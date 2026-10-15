"""Pure NIFTY option-contract selection helpers for the Fyers market-data path."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class FyersOptionContract:
    symbol: str
    expiry: date
    option_type: str
    strike: float
    lot_size: int


def _parse_expiry(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, bool):
        raise ValueError("Expiry must be a date or date-like value.")
    if isinstance(value, (int, float)):
        if not isfinite(float(value)):
            raise ValueError("Expiry timestamp must be finite.")
        return datetime.fromtimestamp(float(value), tz=UTC).date()

    text = str(value).strip()
    if not text:
        raise ValueError("Expiry must not be empty.")
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        for pattern in ("%d-%b-%Y", "%d-%b-%y", "%d-%m-%Y"):
            try:
                return datetime.strptime(text, pattern).date()
            except ValueError:
                continue
    raise ValueError(f"Unrecognized Fyers expiry value: {value!r}")


def select_next_week_expiry(expiries: Iterable[Any], *, today: date) -> date:
    """Select the earliest listed expiry in the next ISO calendar week."""
    next_monday = today + timedelta(days=7 - today.weekday())
    following_monday = next_monday + timedelta(days=7)
    candidates: set[date] = set()

    for item in expiries:
        value = item
        if isinstance(item, dict):
            value = item.get("date") or item.get("expiry")
        if value is None:
            continue
        try:
            expiry = _parse_expiry(value)
        except (OverflowError, OSError, ValueError):
            continue
        if next_monday <= expiry < following_monday:
            candidates.add(expiry)

    if not candidates:
        raise ValueError(f"Fyers returned no listed expiry for the week of {next_monday}.")
    return min(candidates)


def select_parent_style_expiry(expiries: Iterable[Any], *, today: date) -> date:
    """Select the second listed expiry on or after today, like the parent EMA worker."""
    candidates: set[date] = set()
    for item in expiries:
        value = (item.get("date") or item.get("expiry")) if isinstance(item, dict) else item
        if value is None:
            continue
        try:
            expiry = _parse_expiry(value)
        except (OverflowError, OSError, ValueError):
            continue
        if expiry >= today:
            candidates.add(expiry)
    ordered = sorted(candidates)
    if len(ordered) < 2:
        raise ValueError(f"Fyers returned fewer than two listed expiries on or after {today}.")
    return ordered[1]


def select_atm_option(
    options: Iterable[dict[str, Any]],
    *,
    action: str,
    spot: float,
    expiry: date,
    strike_step: float | None = None,
) -> FyersOptionContract:
    """Choose the nearest CE/PE strike; ties resolve to the lower strike."""
    direction = str(action).strip().upper()
    option_type = {"ENTER_LONG": "CE", "LONG": "CE", "ENTER_SHORT": "PE", "SHORT": "PE"}.get(direction)
    if option_type is None:
        raise ValueError(f"Unsupported EMA entry action: {action!r}")
    if isinstance(spot, bool) or not isfinite(float(spot)) or float(spot) <= 0:
        raise ValueError("Underlying spot price must be finite and positive.")
    if strike_step is not None and (
        isinstance(strike_step, bool) or not isfinite(float(strike_step)) or float(strike_step) <= 0
    ):
        raise ValueError("strike_step must be finite and positive when supplied.")
    target_strike = (
        float(round(float(spot) / float(strike_step)) * float(strike_step)) if strike_step is not None else float(spot)
    )

    candidates: list[tuple[float, float, FyersOptionContract]] = []
    for row in options:
        if not isinstance(row, dict):
            continue
        row_type = str(row.get("option_type", "")).strip().upper()
        if row_type != option_type:
            continue
        try:
            strike = float(row.get("strike_price"))
            lot_size_value = float(row.get("lot_size"))
            lot_size = int(lot_size_value)
        except (TypeError, ValueError, OverflowError):
            continue
        symbol = str(row.get("symbol", "")).strip()
        if (
            not symbol
            or not isfinite(strike)
            or strike <= 0
            or not isfinite(lot_size_value)
            or lot_size_value != lot_size
            or lot_size <= 0
        ):
            continue
        contract = FyersOptionContract(symbol, expiry, option_type, strike, lot_size)
        candidates.append((abs(strike - target_strike), strike, contract))

    if not candidates:
        raise ValueError(f"Fyers option chain has no valid {option_type} contracts.")
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]


def select_parent_style_atm_option(
    options: Iterable[dict[str, Any]],
    *,
    action: str,
    spot: float,
    today: date,
) -> FyersOptionContract:
    """Select second-upcoming expiry and parent-style rounded ATM strike."""
    rows = [row for row in options if isinstance(row, dict)]
    expiry = select_parent_style_expiry((row.get("expiry") for row in rows), today=today)
    expiry_rows = []
    for row in rows:
        try:
            row_expiry = _parse_expiry(row.get("expiry"))
        except (OverflowError, OSError, ValueError):
            continue
        if row_expiry == expiry:
            expiry_rows.append(row)
    return select_atm_option(
        expiry_rows,
        action=action,
        spot=spot,
        expiry=expiry,
        strike_step=50,
    )


def select_next_week_atm_option(
    options: Iterable[dict[str, Any]],
    *,
    action: str,
    spot: float,
    today: date,
) -> FyersOptionContract:
    rows = [row for row in options if isinstance(row, dict)]
    expiry = select_next_week_expiry((row.get("expiry") for row in rows), today=today)
    expiry_options = []
    for row in rows:
        try:
            row_expiry = _parse_expiry(row.get("expiry"))
        except (OverflowError, OSError, ValueError):
            continue
        if row_expiry == expiry:
            expiry_options.append(row)
    return select_atm_option(expiry_options, action=action, spot=spot, expiry=expiry)
