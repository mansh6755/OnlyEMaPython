import pytest

from Dependencies.broker_contract import OrderStatus
from Dependencies.fyers_execution import FyersExecutionClient, _broker_state, _unique_order_tag


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [("2", "TRADED"), ("3", "REJECTED"), ("7", "PARTIAL"), ("5", "PENDING")],
)
def test_fyers_numeric_order_statuses_are_normalized(status_code, expected):
    assert _broker_state({"orderNumStatus": status_code}) == expected


def test_fyers_full_fill_requires_terminal_traded_status():
    result = FyersExecutionClient._normalize_row(
        {"id": "order-1", "orderNumStatus": 2, "filledQty": 65, "tradedPrice": 102.5},
        requested_quantity=65,
    )

    assert result.status is OrderStatus.FILLED
    assert result.filled_quantity == 65
    assert result.average_fill_price == 102.5


def test_fyers_partial_fill_is_not_reported_as_rejection():
    result = FyersExecutionClient._normalize_row(
        {"id": "order-2", "orderNumStatus": 8, "filledQty": 20, "tradedPrice": 102.5},
        requested_quantity=65,
    )

    assert result.status is OrderStatus.PARTIAL
    assert result.filled_quantity == 20
    assert result.remaining_quantity == 45


def test_fyers_partial_state_with_full_quantity_stays_ambiguous():
    result = FyersExecutionClient._normalize_row(
        {"id": "order-3", "orderNumStatus": 7, "filledQty": 65},
        requested_quantity=65,
    )

    assert result.status is OrderStatus.UNKNOWN


def test_fyers_order_tags_are_unique_and_within_broker_limit():
    first = _unique_order_tag("EMA-ENTRY")
    second = _unique_order_tag("EMA-ENTRY")

    assert first != second
    assert first.startswith("EMAE")
    assert len(first) <= 20


def test_fyers_adapter_returns_filled_only_after_orderbook_confirmation():
    class FyersResponse:
        def __init__(self):
            self.row = None

        def profile(self):
            return {"s": "ok"}

        def place_market_order(self, symbol, side, quantity, **kwargs):
            self.row = {
                "id": "order-4",
                "symbol": symbol,
                "side": 1 if side == "BUY" else -1,
                "qty": quantity,
                "filledQty": quantity,
                "orderNumStatus": 2,
                "tradedPrice": 125.0,
                "orderTag": kwargs["order_tag"],
            }
            return {"s": "ok", "id": "order-4"}

        def orderbook(self):
            return {"s": "ok", "orderBook": [self.row]}

    client = FyersExecutionClient(FyersResponse())

    result = client.place_market_order("NSE:NIFTY26O1324000CE", "BUY", 65)

    assert result.status is OrderStatus.FILLED
    assert result.order_id == "order-4"
    assert result.average_fill_price == 125.0
