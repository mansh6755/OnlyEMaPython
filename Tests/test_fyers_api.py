import os
from datetime import UTC, datetime

import pytest

from Dependencies.fyers_api import FyersAPI, FyersAPIError, build_auth_url, parse_nse_fo_master, quote_ltp


class _Response:
    def __init__(self, payload, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self.payload


class _Session:
    def __init__(self, response):
        self.response = response
        self.last_request = None

    def request(self, *args, **kwargs):
        self.last_request = (args, kwargs)
        return self.response


def test_quote_ltp_reads_the_requested_fyers_symbol():
    response = {
        "s": "ok",
        "d": [
            {"n": "NSE:OTHER-EQ", "v": {"lp": 12.5}},
            {"n": "NSE:NIFTY50-INDEX", "v": {"lp": 24012.25}},
        ],
    }

    assert quote_ltp(response, "NSE:NIFTY50-INDEX") == 24012.25


@pytest.mark.parametrize(
    "response",
    [{}, {"d": []}, {"d": [{"n": "NSE:NIFTY50-INDEX", "v": {"lp": 0}}]}],
)
def test_quote_ltp_rejects_missing_or_invalid_prices(response):
    with pytest.raises(FyersAPIError):
        quote_ltp(response, "NSE:NIFTY50-INDEX")


def test_build_auth_url_encodes_redirect_and_state():
    url = build_auth_url("app-100", "https://localhost/callback?x=1", "state-1")

    assert "client_id=app-100" in url
    assert "state=state-1" in url
    assert "redirect_uri=https%3A%2F%2Flocalhost%2Fcallback%3Fx%3D1" in url


def test_parse_nse_fo_master_reads_exact_symbol_and_lot_size_columns():
    row = (
        "101126101335028,NIFTY 13 Oct 26 17350 CE,14,65,0.05,,0915-1540:,"
        "2026-10-08,1791886200,NSE:NIFTY26O1317350CE,10,11,35028,NIFTY,26000,"
        "17350.0,CE,10100000026000,None,0,0.0"
    )

    contracts = parse_nse_fo_master(row)

    assert contracts == [
        {
            "symbol": "NSE:NIFTY26O1317350CE",
            "underlying": "NIFTY",
            "expiry": 1791886200,
            "strike_price": 17350.0,
            "option_type": "CE",
            "lot_size": 65,
        }
    ]


def test_fyers_history_converts_epoch_candles_to_ist_ohlcv():
    epoch = int(datetime(2026, 10, 9, 3, 45, tzinfo=UTC).timestamp())
    api = FyersAPI(
        "app-100",
        "test-token",
        session=_Session(_Response({"s": "ok", "candles": [[epoch, 100, 105, 99, 104, 60]]})),
    )

    candles = api.history(
        "NSE:NIFTY50-INDEX",
        resolution="1",
        start=datetime(2026, 10, 9).date(),
        end=datetime(2026, 10, 9).date(),
    )

    assert candles.iloc[0]["timestamp"].isoformat() == "2026-10-09T09:15:00+05:30"
    assert candles.iloc[0]["open"] == 100
    assert candles.iloc[0]["high"] == 105
    assert candles.iloc[0]["low"] == 99
    assert candles.iloc[0]["close"] == 104
    assert candles.iloc[0]["volume"] == 60


def test_option_chain_uses_the_fyers_data_api_base():
    session = _Session(_Response({"s": "ok", "data": {"optionsChain": []}}))
    api = FyersAPI("app-100", "test-token", session=session)

    api.option_chain("NSE:NIFTY50-INDEX", expiry_timestamp="1791886200")

    assert session.last_request[0][1] == "https://api-t1.fyers.in/data/options-chain-v3"
    assert session.last_request[1]["headers"]["version"] == "3"


def test_history_uses_epoch_ranges_for_five_second_candles():
    session = _Session(_Response({"s": "ok", "candles": []}))
    api = FyersAPI("app-100", "test-token", session=session)

    api.history(
        "NSE:NIFTY50-INDEX",
        resolution="5S",
        start=datetime(2026, 10, 1).date(),
        end=datetime(2026, 10, 9).date(),
    )

    args, kwargs = session.last_request
    assert args[1] == "https://api-t1.fyers.in/data/history"
    assert kwargs["params"]["resolution"] == "5S"
    assert kwargs["params"]["date_format"] == "0"
    assert kwargs["params"]["range_from"].isdigit()
    assert kwargs["params"]["range_to"].isdigit()


def test_fyers_get_retries_rate_limit_once(monkeypatch):
    class QueueSession:
        def __init__(self):
            self.responses = [
                _Response({}, status_code=429, headers={"Retry-After": "0"}),
                _Response({"s": "ok", "candles": []}),
            ]
            self.calls = 0

        def request(self, *args, **kwargs):
            self.calls += 1
            return self.responses.pop(0)

    session = QueueSession()
    monkeypatch.setattr("Dependencies.fyers_api.sleep_time.sleep", lambda _delay: None)
    api = FyersAPI("app-100", "test-token", session=session)

    response = api._request("GET", "https://example.test", "/history")

    assert response["s"] == "ok"
    assert session.calls == 2


def test_replace_access_token_persists_and_updates_future_request_headers(monkeypatch):
    session = _Session(_Response({"s": "ok", "candles": []}))
    api = FyersAPI("app-100", "old-token", session=session)
    persisted = []
    monkeypatch.delenv("FYERS_ACCESS_TOKEN", raising=False)

    api.replace_access_token("new-token", persist=persisted.append)
    api._request("GET", "https://example.test", "/profile")

    assert persisted == ["new-token"]
    assert api.access_token == "new-token"
    assert session.last_request[1]["headers"]["Authorization"] == "app-100:new-token"
    assert os.environ["FYERS_ACCESS_TOKEN"] == "new-token"


def test_clear_access_token_removes_active_token_and_environment(monkeypatch):
    api = FyersAPI("app-100", "old-token", session=_Session(_Response({"s": "ok"})))
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", "old-token")

    api.clear_access_token()

    assert api.access_token == ""
    assert api._headers["Authorization"] == "app-100:"
    assert "FYERS_ACCESS_TOKEN" not in os.environ


def test_validate_session_raises_when_fyers_rejects_session():
    api = FyersAPI("app-100", "test-token", session=_Session(_Response({"s": "no"})))

    with pytest.raises(RuntimeError, match="rejected the access token"):
        api.validate_session()


def test_expired_history_uses_fno_expired_endpoint():
    session = _Session(_Response({"s": "ok", "candles": []}))
    api = FyersAPI("app-100", "test-token", session=session)

    api.history(
        "NSE:NIFTY26O0622500CE",
        resolution="5S",
        start=datetime(2026, 10, 1).date(),
        end=datetime(2026, 10, 6).date(),
        expired=True,
    )

    args, kwargs = session.last_request
    assert args[1] == "https://api-t1.fyers.in/data/history/fno/expired/historical-data"
    assert kwargs["params"]["include_greeks"] == "0"


def test_expired_history_accepts_fyers_no_data_response_as_empty():
    session = _Session(_Response({"s": "no_data", "candles": []}))
    api = FyersAPI("app-100", "test-token", session=session)

    candles = api.history(
        "NSE:NIFTY2650517100CE",
        resolution="1",
        start=datetime(2026, 5, 5).date(),
        end=datetime(2026, 5, 5).date(),
        expired=True,
    )

    assert candles.empty
    assert list(candles.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_expired_option_expiries_and_symbols_unwrap_data_payload():
    class QueueSession:
        def __init__(self):
            self.responses = [
                _Response({"s": "ok", "data": {"expiry_dates": {"options": ["2026-10-06"]}}}),
                _Response({"s": "ok", "data": {"contracts": {"options": ["NSE:NIFTY26O0622500CE"]}}}),
            ]

        def request(self, *args, **kwargs):
            return self.responses.pop(0)

    api = FyersAPI("app-100", "test-token", session=QueueSession())

    assert api.expired_option_expiries(
        "NSE:NIFTY50-INDEX", datetime(2026, 10, 1).date(), datetime(2026, 10, 9).date()
    ) == [datetime(2026, 10, 6).date()]
    assert api.expired_option_symbols(
        "NSE:NIFTY50-INDEX", datetime(2026, 10, 6).date()
    ) == ["NSE:NIFTY26O0622500CE"]
