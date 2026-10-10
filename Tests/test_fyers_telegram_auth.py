import socket
from pathlib import Path
from unittest.mock import Mock

import pytest

from Dependencies import fyers_telegram_auth as auth


class _FakeMarketDataClient:
    def __init__(self) -> None:
        self.installed_token: str | None = None
        self.session_valid = True

    def replace_access_token(self, token: str, *, persist) -> None:
        persist(token)
        self.installed_token = token
        self.session_valid = True

    def validate_session(self) -> None:
        if not self.session_valid:
            raise RuntimeError("invalid Fyers session")


def _unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _service(tmp_path: Path) -> auth.FyersTelegramAuthService:
    return auth.FyersTelegramAuthService(
        bot_token="bot-token",
        authorized_chat_id="12345",
        client_id="app-id",
        secret_key="app-secret",
        redirect_uri="https://service.example/callback/",
        token_file=tmp_path / "fyers_access_token",
        market_data_client=_FakeMarketDataClient(),
        port=_unused_port(),
    )


def test_persisted_token_takes_precedence_over_environment_fallback(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("stored-token\n", encoding="utf-8")

    assert auth.load_persisted_access_token(token_file, "environment-token") == "stored-token"
    assert auth.load_persisted_access_token(tmp_path / "missing", "environment-token") == "environment-token"


def test_access_token_is_atomically_persisted_with_private_permissions(tmp_path):
    token_file = tmp_path / "nested" / "token"

    auth.persist_access_token(token_file, "new-token")

    assert token_file.read_text(encoding="utf-8") == "new-token"
    assert token_file.stat().st_mode & 0o777 == 0o600


def test_callback_checks_state_exchanges_code_and_persists_token(tmp_path, monkeypatch):
    service = _service(tmp_path)
    service._pending_state = "expected-state"
    service._pending_state_expires_at = 10**20
    service._notify = Mock()
    exchange = Mock(return_value="new-access-token")
    monkeypatch.setattr(auth, "exchange_auth_code", exchange)

    service._handle_callback(
        "/callback/?s=ok&auth_code=temporary-code&state=expected-state"
    )

    exchange.assert_called_once_with("app-id", "app-secret", "temporary-code")
    assert service.market_data_client.installed_token == "new-access-token"
    assert (tmp_path / "fyers_access_token").read_text(encoding="utf-8") == "new-access-token"
    assert service._token_updated.is_set()
    assert service._pending_state is None
    assert "new-access-token" not in service._notify.call_args.args[0]
    service._httpd.server_close()


def test_callback_rejects_wrong_or_replayed_state(tmp_path, monkeypatch):
    service = _service(tmp_path)
    service._pending_state = "expected-state"
    service._pending_state_expires_at = 10**20
    exchange = Mock()
    monkeypatch.setattr(auth, "exchange_auth_code", exchange)

    with pytest.raises(ValueError, match="state"):
        service._handle_callback("/callback/?auth_code=temporary-code&state=wrong")

    service._pending_state = None
    with pytest.raises(ValueError, match="state"):
        service._handle_callback("/callback/?auth_code=temporary-code&state=expected-state")

    service._httpd.server_close()
    exchange.assert_not_called()


def test_startup_requests_auth_for_missing_or_invalid_token(tmp_path):
    service = _service(tmp_path)
    service._notify = Mock()
    service.market_data_client.session_valid = False
    request_auth = Mock(side_effect=lambda _reason: _make_session_valid(service))
    service.request_authorization = request_auth

    service.authenticate_startup(has_access_token=True)

    request_auth.assert_called_once()
    assert "invalid or expired" in request_auth.call_args.args[0]
    service._httpd.server_close()


def test_startup_accepts_valid_token_without_prompt(tmp_path):
    service = _service(tmp_path)
    request_auth = Mock()
    service.request_authorization = request_auth

    service.authenticate_startup(has_access_token=True)

    request_auth.assert_not_called()
    service._httpd.server_close()


def _make_session_valid(service: auth.FyersTelegramAuthService) -> None:
    service.market_data_client.session_valid = True
    service._token_updated.set()


@pytest.mark.parametrize(
    ("chat_type", "chat_id", "accepted"),
    [("private", 12345, True), ("private", 54321, False), ("group", 12345, False)],
)
def test_only_configured_private_chat_can_request_auth(
    tmp_path,
    chat_type,
    chat_id,
    accepted,
):
    service = _service(tmp_path)
    request_auth = Mock()
    service.request_authorization = request_auth

    service._handle_update(
        {
            "message": {
                "chat": {"id": chat_id, "type": chat_type},
                "text": "/auth@trade_bot",
            }
        }
    )

    assert request_auth.called is accepted
    service._httpd.server_close()


def test_auth_service_requires_https_and_numeric_private_chat(tmp_path):
    common = {
        "bot_token": "bot-token",
        "authorized_chat_id": "12345",
        "client_id": "app-id",
        "secret_key": "app-secret",
        "redirect_uri": "https://service.example/callback/",
        "token_file": tmp_path / "token",
        "market_data_client": _FakeMarketDataClient(),
        "port": _unused_port(),
    }

    with pytest.raises(ValueError, match="HTTPS"):
        auth.FyersTelegramAuthService(**(common | {"redirect_uri": "http://service.example/callback"}))
    with pytest.raises(ValueError, match="numeric private-chat"):
        auth.FyersTelegramAuthService(**(common | {"authorized_chat_id": "@channel"}))
