from datetime import UTC, datetime

import pytest

from Dependencies.fyers_api import FyersAPI, FyersAPIError, build_auth_url, parse_nse_fo_master, quote_ltp


class _Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

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
