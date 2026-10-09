"""Fyers order adapter implementing the repository's broker-neutral contract."""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from Dependencies.broker_contract import (
    BrokerQueryResult,
    OpenOrder,
    OpenPosition,
    OrderResult,
    OrderStatus,
    exact_int,
    normalize_order_result,
)
from Dependencies.fyers_api import FyersAPI, FyersAPIError
from Dependencies.fyers_options import _parse_expiry

DEPENDENCIES_DIR = Path(__file__).resolve().parent
load_dotenv(DEPENDENCIES_DIR / ".env")

_FILL_STATES = {"2": "TRADED", "TRADED": "TRADED", "FILLED": "FILLED", "COMPLETE": "COMPLETE"}
_REJECT_STATES = {
    "1": "CANCELLED",
    "3": "REJECTED",
    "6": "CANCELLED",
    "8": "CANCELLED",
    "CANCELLED": "CANCELLED",
    "CANCELED": "CANCELLED",
    "REJECTED": "REJECTED",
    "EXPIRED": "CANCELLED",
}
_PARTIAL_STATES = {"7": "PARTIAL", "PARTIAL": "PARTIAL", "PARTIALLY EXECUTED": "PARTIAL"}
_TRANSIENT_STATES = {"4": "TRANSIT", "5": "PENDING", "TRANSIT": "TRANSIT", "PENDING": "PENDING"}


def _broker_state(row: dict[str, Any]) -> str:
    raw = row.get("orderNumStatus", row.get("status", ""))
    value = str(raw).strip().upper()
    return (
        _FILL_STATES.get(value)
        or _REJECT_STATES.get(value)
        or _PARTIAL_STATES.get(value)
        or _TRANSIENT_STATES.get(value)
        or value
    )


def _order_rows(response: dict[str, Any]) -> list[dict[str, Any]] | None:
    rows = response.get("orderBook")
    if rows is None and isinstance(response.get("data"), dict):
        rows = response["data"].get("orderBook")
    if not isinstance(rows, list):
        return None
    return [row for row in rows if isinstance(row, dict)]


def _unique_order_tag(prefix: str) -> str:
    normalized_prefix = "".join(character for character in prefix.upper() if character.isalnum())[:4]
    return f"{normalized_prefix or 'EMA'}{uuid.uuid4().hex[:16]}"[:20]


class FyersExecutionClient:
    """One lazily authenticated Fyers session with serialized order outcomes."""

    def __init__(self, api: FyersAPI | None = None) -> None:
        self.api = api
        self.is_logged_in = False
        self._lock = threading.RLock()
        self._order_lock = threading.Lock()
        self._blocked = False
        self._scrip_master: list[dict[str, Any]] | None = None

    def _client(self) -> FyersAPI:
        if self.api is not None:
            return self.api
        self.api = FyersAPI(
            os.getenv("FYERS_APP_ID", ""),
            os.getenv("FYERS_ACCESS_TOKEN", ""),
        )
        return self.api

    def ensure_logged_in(self) -> bool:
        with self._lock:
            if self.is_logged_in:
                return True
            try:
                profile = self._client().profile()
            except (FyersAPIError, ValueError):
                return False
            self.is_logged_in = profile.get("s") == "ok"
            return self.is_logged_in

    def preload_scrip_master(self) -> bool:
        if not self.ensure_logged_in():
            return False
        try:
            self._scrip_master = self._client().option_master()
        except FyersAPIError:
            return False
        return bool(self._scrip_master)

    def resolve_option_symbol(
        self,
        underlying: str,
        expiry: Any,
        option_type: str,
        strike: float,
        exchange_segment: str = "NSE_FNO",
    ) -> str:
        if str(exchange_segment).strip().upper() not in {"NSE_FNO", "NSE"}:
            return ""
        if not self.ensure_logged_in():
            return ""
        if self._scrip_master is None and not self.preload_scrip_master():
            return ""
        try:
            expiry_date = _parse_expiry(expiry)
            strike_value = float(strike)
        except (TypeError, ValueError, OverflowError, OSError):
            return ""
        for row in self._scrip_master or []:
            try:
                matches = (
                    row["underlying"] == str(underlying).strip().upper()
                    and row["option_type"] == str(option_type).strip().upper()
                    and _parse_expiry(row["expiry"]) == expiry_date
                    and float(row["strike_price"]) == strike_value
                )
            except (KeyError, TypeError, ValueError, OverflowError, OSError):
                continue
            if matches:
                return str(row.get("symbol", "")).strip()
        return ""

    @staticmethod
    def _normalize_row(row: dict[str, Any], requested_quantity: int) -> OrderResult:
        state = _broker_state(row)
        filled = row.get("filledQty", row.get("filled_qty", row.get("tradedQty", 0)))
        filled_quantity = exact_int(filled)
        if state == "PARTIAL" and filled_quantity == requested_quantity:
            state = "UNKNOWN"
        price = row.get("tradedPrice", row.get("avgPrice", row.get("averagePrice", 0)))
        return normalize_order_result(
            order_id=row.get("id", row.get("orderId", "")),
            requested_quantity=requested_quantity,
            filled_quantity=filled,
            broker_state=state,
            reason="Fyers order-book confirmation.",
            average_fill_price=price,
        )

    def _find_order(self, order_id: str = "", order_tag: str = "") -> dict[str, Any] | None:
        rows = _order_rows(self._client().orderbook())
        if rows is None:
            return None
        for row in rows:
            if order_id and str(row.get("id", row.get("orderId", ""))) == order_id:
                return row
            if order_tag and str(row.get("orderTag", "")) == order_tag:
                return row
        return None

    def _wait_for_order(
        self,
        order_id: str,
        order_tag: str,
        requested_quantity: int,
        *,
        deadline: float,
    ) -> OrderResult:
        last_row: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            try:
                row = self._find_order(order_id, order_tag)
            except FyersAPIError:
                row = None
            if row is not None:
                last_row = row
                result = self._normalize_row(row, requested_quantity)
                if result.status is not OrderStatus.UNKNOWN or result.filled_quantity > 0:
                    if result.status is not OrderStatus.FILLED:
                        self._blocked = True
                    return result
                if result.broker_state in {"REJECTED", "CANCELLED"}:
                    return result
            time.sleep(min(0.4, max(0.0, deadline - time.monotonic())))
        if last_row is not None:
            self._blocked = True
            return self._normalize_row(last_row, requested_quantity)
        self._blocked = True
        return normalize_order_result(
            order_id=order_id,
            requested_quantity=requested_quantity,
            filled_quantity=0,
            broker_state="UNKNOWN",
            reason="Fyers did not provide an order-book outcome before the deadline.",
        )

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        exchange_segment: str = "NSE_FNO",
        product_type: str = "INTRADAY",
        *,
        order_tag: str = "",
    ) -> OrderResult:
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("quantity must be a positive integer")
        if not self._order_lock.acquire(timeout=10.0):
            return normalize_order_result(
                order_id="",
                requested_quantity=quantity,
                filled_quantity=0,
                broker_state="UNKNOWN",
                reason="Another Fyers order is unresolved; no new order was submitted.",
            )
        try:
            if self._blocked:
                return normalize_order_result(
                    order_id="",
                    requested_quantity=quantity,
                    filled_quantity=0,
                    broker_state="UNKNOWN",
                    reason="Fyers submissions are blocked pending account reconciliation.",
                )
            if not self.ensure_logged_in():
                return normalize_order_result(
                    order_id="",
                    requested_quantity=quantity,
                    filled_quantity=0,
                    broker_state="REJECTED",
                    reason="Fyers session is unavailable; the order was not submitted.",
                )
            tag = _unique_order_tag(order_tag)
            try:
                response = self._client().place_market_order(
                    symbol,
                    side,
                    quantity,
                    product_type=product_type,
                    order_tag=tag,
                )
            except FyersAPIError:
                return self._wait_for_order("", tag, quantity, deadline=time.monotonic() + 8.0)
            order_id = str(response.get("id", response.get("orderId", ""))).strip()
            response_status = str(response.get("s", "")).lower()
            if not order_id and response_status != "error":
                return self._wait_for_order("", tag, quantity, deadline=time.monotonic() + 8.0)
            if not order_id:
                return normalize_order_result(
                    order_id="",
                    requested_quantity=quantity,
                    filled_quantity=0,
                    broker_state="REJECTED",
                    reason="Fyers rejected the order before assigning an order id.",
                )
            return self._wait_for_order(order_id, tag, quantity, deadline=time.monotonic() + 8.0)
        finally:
            self._order_lock.release()

    def get_order_status(self, order_id: str, requested_quantity: int = 0) -> OrderResult:
        try:
            row = self._find_order(order_id)
        except FyersAPIError:
            row = None
        if row is None:
            return normalize_order_result(
                order_id=order_id,
                requested_quantity=requested_quantity,
                filled_quantity=0,
                broker_state="UNKNOWN",
                reason="Fyers order book did not confirm this order.",
            )
        requested = exact_int(row.get("qty", row.get("requestedQty", requested_quantity)))
        if requested is None or requested <= 0:
            requested = requested_quantity
        return self._normalize_row(row, requested)

    def cancel_order(self, order_id: str, requested_quantity: int = 0) -> OrderResult:
        try:
            self._client().cancel_order(order_id)
        except (FyersAPIError, ValueError):
            self._blocked = True
        return self.get_order_status(order_id, requested_quantity)

    def list_open_orders(self) -> BrokerQueryResult[OpenOrder]:
        try:
            rows = _order_rows(self._client().orderbook())
        except FyersAPIError as exc:
            return BrokerQueryResult.indeterminate(str(exc), broker_state="FYERS_ORDERBOOK_ERROR")
        if rows is None:
            return BrokerQueryResult.indeterminate(
                "Fyers order book did not include its order list.", broker_state="MALFORMED"
            )
        open_orders: list[OpenOrder] = []
        for row in rows:
            state = _broker_state(row)
            requested = exact_int(row.get("qty", row.get("requestedQty")))
            filled = exact_int(row.get("filledQty", row.get("filled_qty", 0)))
            if requested is None or filled is None or requested < 0 or filled < 0 or filled > requested:
                return BrokerQueryResult.indeterminate(
                    "Fyers order book contained malformed quantities.", broker_state="MALFORMED"
                )
            if state in _TRANSIENT_STATES.values() or (state == "PARTIAL" and filled < requested):
                side = row.get("side", "")
                side_text = "BUY" if str(side) == "1" else "SELL" if str(side) == "-1" else str(side)
                open_orders.append(
                    OpenOrder(
                        order_id=str(row.get("id", row.get("orderId", ""))),
                        symbol=str(row.get("symbol", "")),
                        side=side_text,
                        requested_quantity=requested,
                        filled_quantity=filled,
                        remaining_quantity=requested - filled,
                        broker_state=state,
                    )
                )
        return BrokerQueryResult.success(open_orders)

    def list_open_positions(self) -> BrokerQueryResult[OpenPosition]:
        try:
            response = self._client().positions()
        except FyersAPIError as exc:
            return BrokerQueryResult.indeterminate(str(exc), broker_state="FYERS_POSITIONS_ERROR")
        rows = response.get("netPositions")
        if rows is None and isinstance(response.get("data"), dict):
            rows = response["data"].get("netPositions")
        if not isinstance(rows, list):
            return BrokerQueryResult.indeterminate(
                "Fyers positions response did not include its position list.", broker_state="MALFORMED"
            )
        positions: list[OpenPosition] = []
        for row in rows:
            if not isinstance(row, dict):
                return BrokerQueryResult.indeterminate(
                    "Fyers positions response contained a malformed row.", broker_state="MALFORMED"
                )
            quantity = exact_int(row.get("netQty", row.get("net_quantity")))
            if quantity is None:
                return BrokerQueryResult.indeterminate(
                    "Fyers positions response contained an invalid net quantity.", broker_state="MALFORMED"
                )
            if quantity:
                positions.append(
                    OpenPosition(
                        symbol=str(row.get("symbol", "")),
                        quantity=quantity,
                        product_type=str(row.get("productType", row.get("product_type", ""))),
                    )
                )
        return BrokerQueryResult.success(positions)

    def recover_after_reconciliation(self) -> bool:
        orders = self.list_open_orders()
        positions = self.list_open_positions()
        if orders.is_indeterminate or positions.is_indeterminate or orders.items or positions.items:
            return False
        self._blocked = False
        return True

    @staticmethod
    def extract_order_id(order_response: Any) -> str:
        if not isinstance(order_response, dict):
            return ""
        return str(order_response.get("id", order_response.get("orderId", ""))).strip()

    def logout(self) -> dict[str, Any]:
        self.is_logged_in = False
        self.api = None
        self._scrip_master = None
        return {"s": "ok"}


fyers_execution_client = FyersExecutionClient()
