import http.client
import json
import os
import socket
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from Dependencies import fyers_telegram_auth as auth

IST = ZoneInfo("Asia/Kolkata")


class _FakeMarketDataClient:
    def __init__(self) -> None:
        self.installed_token: str | None = None
        self.active_token = "active-token"
        self.session_valid = True

    def clear_access_token(self) -> None:
        self.active_token = ""
        os.environ.pop("FYERS_ACCESS_TOKEN", None)

    def replace_access_token(self, token: str, *, persist) -> None:
        persist(token)
        self.installed_token = token
        self.active_token = token
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


def test_daily_authentication_is_scheduled_at_nine_ist_on_weekdays():
    before_nine = datetime(2026, 10, 22, 8, 59, tzinfo=IST)
    at_nine = datetime(2026, 10, 22, 9, 0, tzinfo=IST)

    assert auth.next_daily_authentication_at(before_nine) == datetime(
        2026, 10, 22, 9, 0, tzinfo=IST
    )
    assert auth.next_daily_authentication_at(at_nine) == datetime(
        2026, 10, 23, 9, 0, tzinfo=IST
    )


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 10, 23, 9, 1, tzinfo=IST),
        datetime(2026, 10, 24, 8, 0, tzinfo=IST),
        datetime(2026, 10, 25, 12, 0, tzinfo=IST),
    ],
)
def test_daily_authentication_skips_weekends(now):
    assert auth.next_daily_authentication_at(now) == datetime(
        2026, 10, 26, 9, 0, tzinfo=IST
    )


def test_daily_authentication_scheduler_requests_a_fresh_link(monkeypatch, tmp_path):
    service = _service(tmp_path)
    request_auth = Mock(side_effect=lambda _reason: service._stop_event.set())
    service.request_authorization = request_auth
    monkeypatch.setattr(
        auth,
        "next_daily_authentication_at",
        lambda _now: datetime.now(IST),
    )

    service._schedule_daily_authentication()

    request_auth.assert_called_once()
    assert "Daily Fyers authentication" in request_auth.call_args.args[0]
    service._httpd.server_close()


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


def test_clear_saved_access_tokens_removes_token_file_and_env_entry(tmp_path):
    token_file = tmp_path / "persistent-token"
    token_file.write_text("old-token", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FYERS_APP_ID=app-id\nFYERS_ACCESS_TOKEN=old-token\nOTHER_SETTING=keep\n",
        encoding="utf-8",
    )

    auth.clear_saved_access_tokens(token_file, env_file)

    assert not token_file.exists()
    assert env_file.read_text(encoding="utf-8") == "FYERS_APP_ID=app-id\nOTHER_SETTING=keep\n"


def test_authorization_clears_saved_and_active_token_before_notifying(tmp_path, monkeypatch):
    service = _service(tmp_path)
    token_file = tmp_path / "fyers_access_token"
    token_file.write_text("old-token", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("FYERS_ACCESS_TOKEN=old-token\n", encoding="utf-8")
    monkeypatch.setattr(auth, "ENV_FILE", env_file)
    monkeypatch.setenv("FYERS_ACCESS_TOKEN", "old-token")
    service.token_file = token_file
    notifications = []

    def capture_notification(message: str) -> None:
        notifications.append(message)
        assert not token_file.exists()
        assert "FYERS_ACCESS_TOKEN" not in env_file.read_text(encoding="utf-8")
        assert "FYERS_ACCESS_TOKEN" not in os.environ
        assert service.market_data_client.active_token == ""

    service._notify = capture_notification

    service.request_authorization("Daily Fyers authentication")

    assert len(notifications) == 1
    service._httpd.server_close()


def test_authorization_does_not_notify_when_token_cleanup_fails(tmp_path, monkeypatch):
    service = _service(tmp_path)
    notify = Mock()
    service._notify = notify
    monkeypatch.setattr(
        auth,
        "clear_saved_access_tokens",
        Mock(side_effect=OSError("permission denied")),
    )

    with pytest.raises(RuntimeError, match="Could not clear saved Fyers tokens"):
        service.request_authorization("Daily Fyers authentication")

    notify.assert_not_called()
    assert service.market_data_client.active_token == "active-token"
    service._httpd.server_close()


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


def test_dashboard_is_public_read_only_when_enabled(tmp_path):
    class _DashboardState:
        def snapshot(self):
            return {"service": {"status": "RUNNING"}, "position": None}

    service = auth.FyersTelegramAuthService(
        bot_token="bot-token",
        authorized_chat_id="12345",
        client_id="app-id",
        secret_key="app-secret",
        redirect_uri="https://service.example/callback/",
        token_file=tmp_path / "token",
        market_data_client=_FakeMarketDataClient(),
        port=_unused_port(),
        dashboard_state=_DashboardState(),
        dashboard_enabled=True,
    )
    server_thread = threading.Thread(target=service._httpd.serve_forever, daemon=True)
    server_thread.start()

    def request(path, *, method="GET", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", service.port, timeout=3)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        body = response.read()
        status = response.status
        connection.close()
        return status, body

    try:
        page_status, page = request("/")
        api_status, api_body = request("/api/dashboard")
        post_status, _ = request("/api/dashboard", method="POST")

        assert page_status == 200
        assert b"NIGHTWATCH" in page
        assert api_status == 200
        assert json.loads(api_body) == {"service": {"status": "RUNNING"}, "position": None}
        assert post_status == 405
    finally:
        service._httpd.shutdown()
        service._httpd.server_close()
        server_thread.join(timeout=2)
