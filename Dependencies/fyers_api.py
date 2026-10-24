"""Timeout-bounded Fyers API v3 client used by the EMA runner."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import time as sleep_time
from collections.abc import Callable
from datetime import date, datetime, time
from io import StringIO
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from Dependencies.fyers_options import (
    FyersOptionContract,
    select_nearest_atm_option,
    select_next_week_atm_option,
)

API_BASE = "https://api-t1.fyers.in/api/v3"
DATA_BASE = "https://api-t1.fyers.in/data"
IST = ZoneInfo("Asia/Kolkata")
NSE_FO_MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"
BSE_FO_MASTER_URL = "https://public.fyers.in/sym_details/BSE_FO.csv"


class FyersAPIError(RuntimeError):
    """A Fyers request failed or returned an unusable response."""


def quote_ltp(response: dict[str, Any], symbol: str) -> float:
    entries = response.get("d")
    if not isinstance(entries, list):
        raise FyersAPIError("Fyers quote response did not contain quote data.")
    for entry in entries:
        if not isinstance(entry, dict) or str(entry.get("n", "")).strip() != symbol:
            continue
        values = entry.get("v")
        if not isinstance(values, dict):
            break
        try:
            price = float(values.get("lp"))
        except (TypeError, ValueError):
            break
        if not math.isfinite(price) or price <= 0:
            break
        return price
    raise FyersAPIError(f"Fyers did not return a valid LTP for {symbol}.")


def build_auth_url(app_id: str, redirect_uri: str, state: str) -> str:
    params = {
        "client_id": app_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "state": state,
    }
    return f"{API_BASE}/generate-authcode?{urlencode(params)}"


def _history_epoch(value: date | datetime, *, end_of_day: bool) -> int:
    if isinstance(value, datetime):
        timestamp = pd.Timestamp(value)
        timestamp = (
            timestamp.tz_localize(IST) if timestamp.tzinfo is None else timestamp.tz_convert(IST)
        )
    else:
        boundary_time = time(23, 59, 59) if end_of_day else time(0, 0)
        timestamp = pd.Timestamp(datetime.combine(value, boundary_time), tz=IST)
    return int(timestamp.timestamp())


def _unwrap_data(response: dict[str, Any]) -> dict[str, Any]:
    data = response.get("data")
    return data if isinstance(data, dict) else response


def parse_fyers_fo_master(content: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fields in csv.reader(StringIO(content)):
        if len(fields) < 17:
            continue
        try:
            lot_size = int(fields[3])
            expiry_timestamp = int(fields[8])
            strike = float(fields[15])
        except (TypeError, ValueError, OverflowError):
            continue
        symbol = fields[9].strip()
        underlying = fields[13].strip().upper()
        option_type = fields[16].strip().upper()
        if lot_size <= 0 or not symbol or not underlying or option_type not in {"CE", "PE"}:
            continue
        rows.append(
            {
                "symbol": symbol,
                "underlying": underlying,
                "expiry": expiry_timestamp,
                "strike_price": strike,
                "option_type": option_type,
                "lot_size": lot_size,
            }
        )
    return rows


def parse_nse_fo_master(content: str) -> list[dict[str, Any]]:
    """Compatibility wrapper for callers using the original NSE parser name."""
    return parse_fyers_fo_master(content)


def exchange_auth_code(
    app_id: str,
    secret_key: str,
    auth_code: str,
    *,
    timeout: tuple[float, float] = (3.0, 10.0),
    session: requests.Session | None = None,
) -> str:
    app_hash = hashlib.sha256(f"{app_id}:{secret_key}".encode()).hexdigest()
    client = session or requests.Session()
    try:
        response = client.post(
            f"{API_BASE}/validate-authcode",
            headers={"Content-Type": "application/json"},
            json={"grant_type": "authorization_code", "appIdHash": app_hash, "code": auth_code},
            timeout=timeout,
        )
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise FyersAPIError("Fyers authorization exchange failed.") from exc
    if response.status_code >= 400 or not isinstance(payload, dict) or payload.get("s") != "ok":
        raise FyersAPIError("Fyers rejected the authorization code.")
    token = str(payload.get("access_token", "")).strip()
    if not token:
        raise FyersAPIError("Fyers authorization response did not include an access token.")
    return token


class FyersAPI:
    def __init__(
        self,
        app_id: str,
        access_token: str,
        *,
        timeout: tuple[float, float] = (3.0, 10.0),
        session: requests.Session | None = None,
    ) -> None:
        self.app_id = str(app_id).strip()
        self.access_token = str(access_token).strip()
        if not self.app_id or not self.access_token:
            raise ValueError("FYERS_APP_ID and FYERS_ACCESS_TOKEN are required.")
        self.timeout = timeout
        self.session = session or requests.Session()
        self._headers = {
            "Authorization": f"{self.app_id}:{self.access_token}",
            "Content-Type": "application/json",
            "version": "3",
        }
        self._option_master_cache: dict[str, list[dict[str, Any]]] = {}
        self._last_get_started = 0.0

    def replace_access_token(self, token: str, *, persist: Callable[[str], None]) -> None:
        refreshed_token = str(token).strip()
        if not refreshed_token:
            raise ValueError("A Fyers access token must not be empty.")
        persist(refreshed_token)
        self.access_token = refreshed_token
        self._headers = {
            **self._headers,
            "Authorization": f"{self.app_id}:{refreshed_token}",
        }
        os.environ["FYERS_ACCESS_TOKEN"] = refreshed_token

    def validate_session(self) -> None:
        profile = self.profile()
        if profile.get("s") != "ok":
            raise RuntimeError("Fyers rejected the access token.")

    def _request(
        self,
        method: str,
        base_url: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = None
        for attempt in range(6):
            if method.upper() == "GET":
                wait_seconds = 0.5 - (sleep_time.monotonic() - self._last_get_started)
                if wait_seconds > 0:
                    sleep_time.sleep(wait_seconds)
                self._last_get_started = sleep_time.monotonic()
            try:
                response = self.session.request(
                    method,
                    f"{base_url}{path}",
                    headers=self._headers,
                    params=params,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise FyersAPIError(f"Fyers {path} request failed.") from exc
            if response.status_code != 429 or method.upper() != "GET" or attempt == 5:
                break
            try:
                retry_after = float(response.headers.get("Retry-After", ""))
            except (TypeError, ValueError):
                retry_after = 0.0
            sleep_time.sleep(min(max(retry_after, 2.0**attempt), 60.0))
        if response is None:
            raise FyersAPIError(f"Fyers {path} request failed without a response.")
        if response.status_code >= 400:
            raise FyersAPIError(f"Fyers {path} returned HTTP {response.status_code}.")
        try:
            result = response.json()
        except ValueError as exc:
            raise FyersAPIError(f"Fyers {path} returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise FyersAPIError(f"Fyers {path} returned a non-object response.")
        if result.get("s") == "error":
            raise FyersAPIError(f"Fyers {path} rejected the request.")
        return result

    def profile(self) -> dict[str, Any]:
        return self._request("GET", API_BASE, "/profile")

    def history(
        self,
        symbol: str,
        *,
        resolution: str,
        start: date | datetime,
        end: date | datetime,
        expired: bool = False,
    ) -> pd.DataFrame:
        if _history_epoch(start, end_of_day=False) > _history_epoch(end, end_of_day=True):
            raise ValueError("history start must not be after end")
        endpoint = "/history/fno/expired/historical-data" if expired else "/history"
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "0",
            "range_from": str(_history_epoch(start, end_of_day=False)),
            "range_to": str(_history_epoch(end, end_of_day=True)),
            "cont_flag": "1",
        }
        if expired:
            params.update({"include_oi": "0", "include_greeks": "0"})
        response = self._request(
            "GET",
            DATA_BASE,
            endpoint,
            params=params,
        )
        candles = _unwrap_data(response).get("candles")
        if response.get("s") == "no_data":
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        if response.get("s") != "ok" or not isinstance(candles, list):
            raise FyersAPIError("Fyers history response did not contain candle data.")
        rows: list[dict[str, Any]] = []
        for candle in candles:
            if not isinstance(candle, (list, tuple)) or len(candle) < 6:
                continue
            try:
                timestamp = pd.to_datetime(candle[0], unit="s", utc=True).tz_convert(IST)
                prices = [float(value) for value in candle[1:5]]
                volume = float(candle[5])
            except (TypeError, ValueError, OverflowError):
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": prices[0],
                    "high": prices[1],
                    "low": prices[2],
                    "close": prices[3],
                    "volume": volume,
                }
            )
        return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

    def expired_option_expiries(self, underlying_symbol: str, start: date, end: date) -> list[date]:
        response = self._request(
            "GET",
            DATA_BASE,
            "/history/fno/expired/expiry-dates",
            params={
                "symbol": underlying_symbol,
                "date_format": "1",
                "range_from": start.isoformat(),
                "range_to": end.isoformat(),
            },
        )
        expiry_dates = _unwrap_data(response).get("expiry_dates", {})
        if not isinstance(expiry_dates, dict):
            raise FyersAPIError("Fyers expired-expiry response was malformed.")
        options = expiry_dates.get("options", [])
        if not isinstance(options, list):
            raise FyersAPIError("Fyers expired option expiry list was malformed.")
        return [date.fromisoformat(str(item)[:10]) for item in options]

    def expired_option_symbols(self, underlying_symbol: str, expiry: date) -> list[str]:
        response = self._request(
            "GET",
            DATA_BASE,
            "/history/fno/expired/underlying-symbols",
            params={"symbol": underlying_symbol, "expiry_date": expiry.isoformat()},
        )
        contracts = _unwrap_data(response).get("contracts", {})
        if not isinstance(contracts, dict):
            raise FyersAPIError("Fyers expired option symbol response was malformed.")
        options = contracts.get("options", [])
        if not isinstance(options, list):
            raise FyersAPIError("Fyers expired option symbols were malformed.")
        return [str(symbol).strip() for symbol in options if str(symbol).strip()]

    def quotes(self, symbols: list[str]) -> dict[str, Any]:
        if not symbols:
            raise ValueError("At least one Fyers quote symbol is required.")
        return self._request(
            "GET",
            DATA_BASE,
            "/quotes",
            params={"symbols": ",".join(symbols)},
        )

    def quote_ltp(self, symbol: str) -> float:
        return quote_ltp(self.quotes([symbol]), symbol)

    def option_chain(
        self,
        symbol: str,
        *,
        expiry_timestamp: str = "",
        strike_count: int = 10,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            DATA_BASE,
            "/options-chain-v3",
            params={
                "symbol": symbol,
                "timestamp": expiry_timestamp,
                "strikecount": strike_count,
                "greeks": "0",
            },
        )

    def option_master(self, exchange: str = "NSE") -> list[dict[str, Any]]:
        exchange_code = str(exchange).strip().upper()
        master_urls = {"NSE": NSE_FO_MASTER_URL, "BSE": BSE_FO_MASTER_URL}
        if exchange_code not in master_urls:
            raise ValueError("exchange must be NSE or BSE.")
        if exchange_code in self._option_master_cache:
            return self._option_master_cache[exchange_code]
        try:
            response = self.session.get(master_urls[exchange_code], timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FyersAPIError(f"Unable to download the Fyers {exchange_code} derivatives master.") from exc

        rows = parse_fyers_fo_master(response.text)
        if not rows:
            raise FyersAPIError(f"Fyers {exchange_code} derivatives master contained no usable contracts.")
        self._option_master_cache[exchange_code] = rows
        return rows

    def resolve_next_week_option(
        self,
        action: str,
        *,
        spot: float,
        today: date,
        underlying: str = "NIFTY",
    ) -> FyersOptionContract:
        options = [
            row
            for row in self.option_master()
            if row["underlying"] == underlying.upper()
            and row["option_type"] in {"CE", "PE"}
        ]
        return select_next_week_atm_option(options, action=action, spot=spot, today=today)

    def resolve_nearest_option(
        self,
        action: str,
        *,
        spot: float,
        today: date,
        underlying: str = "SENSEX",
        exchange: str = "BSE",
    ) -> FyersOptionContract:
        exchange_code = str(exchange).strip().upper()
        underlying_code = str(underlying).strip().upper()
        options = [
            row
            for row in self.option_master(exchange_code)
            if row["underlying"] == underlying_code and row["option_type"] in {"CE", "PE"}
        ]
        return select_nearest_atm_option(options, action=action, spot=spot, today=today)

    def positions(self) -> dict[str, Any]:
        return self._request("GET", API_BASE, "/positions")

    def orderbook(self) -> dict[str, Any]:
        return self._request("GET", API_BASE, "/orders")

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        *,
        product_type: str = "INTRADAY",
        order_tag: str = "",
    ) -> dict[str, Any]:
        side_value = {"BUY": 1, "SELL": -1}.get(str(side).strip().upper())
        if side_value is None:
            raise ValueError("side must be BUY or SELL")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("quantity must be a positive integer")
        body: dict[str, Any] = {
            "symbol": symbol,
            "qty": quantity,
            "type": 2,
            "side": side_value,
            "productType": product_type,
            "limitPrice": 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "isSliceOrder": False,
        }
        if order_tag:
            body["orderTag"] = order_tag[:20]
        return self._request("POST", API_BASE, "/orders/sync", payload=body)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        if not str(order_id).strip():
            raise ValueError("order_id must not be empty")
        return self._request("DELETE", API_BASE, "/orders/sync", payload={"id": str(order_id)})
