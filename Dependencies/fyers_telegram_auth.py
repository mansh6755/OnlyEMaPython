"""Telegram-triggered Fyers OAuth with a guarded HTTPS callback endpoint."""

from __future__ import annotations

import logging
import os
import secrets
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlsplit

import requests

from Dependencies.fyers_api import FyersAPI, FyersAPIError, build_auth_url, exchange_auth_code

LOGGER = logging.getLogger(__name__)
TELEGRAM_API = "https://api.telegram.org"
AUTH_CODE_TTL_SECONDS = 10 * 60
SESSION_CHECK_INTERVAL_SECONDS = 5 * 60
HTTP_TIMEOUT_SECONDS = (5.0, 30.0)


def load_persisted_access_token(token_file: Path, fallback: str = "") -> str:
    """Prefer the token on a persistent volume, otherwise use the env value."""
    if not token_file.exists():
        return fallback.strip()
    token = token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"Persisted Fyers token file is empty: {token_file}")
    return token


def persist_access_token(token_file: Path, access_token: str) -> None:
    """Atomically store a refreshed token with owner-only permissions."""
    token_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=token_file.parent,
            prefix=f".{token_file.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(access_token)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            temporary_path.chmod(0o600)
        except OSError:
            LOGGER.warning("Could not restrict permissions on the persisted Fyers token file.")
        os.replace(temporary_path, token_file)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class FyersTelegramAuthService:
    """Poll one private Telegram chat and serve the Fyers OAuth redirect."""

    def __init__(
        self,
        *,
        bot_token: str,
        authorized_chat_id: str,
        client_id: str,
        secret_key: str,
        redirect_uri: str,
        token_file: Path,
        market_data_client: FyersAPI,
        port: int,
    ) -> None:
        parsed_redirect = urlparse(redirect_uri)
        callback_path = parsed_redirect.path.rstrip("/") or "/"
        if (
            parsed_redirect.scheme != "https"
            or not parsed_redirect.netloc
            or parsed_redirect.query
            or parsed_redirect.fragment
        ):
            raise ValueError("FYERS_REDIRECT_URI must be an HTTPS URL without query or fragment.")
        if callback_path == "/":
            raise ValueError("FYERS_REDIRECT_URI must use a dedicated callback path, not `/`.")
        if not authorized_chat_id.strip().isdigit() or int(authorized_chat_id.strip()) <= 0:
            raise ValueError("FYERS_AUTH_TELEGRAM_CHAT_ID must be a numeric private-chat ID.")
        if not bot_token.strip() or not client_id.strip() or not secret_key.strip():
            raise ValueError("Fyers Telegram authentication credentials are incomplete.")
        if not 1 <= port <= 65535:
            raise ValueError("Railway PORT must be between 1 and 65535.")

        self.bot_token = bot_token.strip()
        self.authorized_chat_id = authorized_chat_id.strip()
        self.client_id = client_id.strip()
        self.secret_key = secret_key.strip()
        self.redirect_uri = redirect_uri
        self.callback_path = callback_path
        self.token_file = token_file
        self.market_data_client = market_data_client
        self.port = port
        self._stop_event = threading.Event()
        self._token_updated = threading.Event()
        self._state_lock = threading.Lock()
        self._pending_state: str | None = None
        self._pending_state_expires_at = 0.0
        self._offset = 0
        self._updates_initialized = False
        self._auth_prompted = False
        self._next_session_check_at = time.monotonic() + SESSION_CHECK_INTERVAL_SECONDS
        self._poll_thread: threading.Thread | None = None
        self._httpd = self._build_http_server()
        self.port = self._httpd.server_port
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="FyersOAuthCallback",
            daemon=True,
        )

    def _build_http_server(self) -> ThreadingHTTPServer:
        """Create a callback server without access-log leakage of OAuth codes."""
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def _respond(self, status: int, message: str) -> None:
                body = message.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                request_path = urlsplit(self.path).path.rstrip("/") or "/"
                if request_path == "/":
                    self._respond(200, "Fyers authentication callback is ready.")
                    return
                if request_path != service.callback_path:
                    self._respond(404, "Not found.")
                    return
                try:
                    service._handle_callback(self.path)
                except (FyersAPIError, OSError, requests.RequestException, RuntimeError, ValueError) as exc:
                    LOGGER.warning("Fyers OAuth callback failed (%s).", type(exc).__name__)
                    service._notify(
                        "Fyers authentication did not complete. Send /auth to this bot to try again."
                    )
                    self._respond(400, "Authentication was not completed. Return to Telegram and retry.")
                    return
                self._respond(200, "Fyers authentication completed. You can return to Telegram.")

            def do_POST(self) -> None:
                self._respond(405, "Method not allowed.")

        return ThreadingHTTPServer(("0.0.0.0", self.port), Handler)

    def start(self) -> None:
        """Start the public callback endpoint and private-chat command poller."""
        self._http_thread.start()
        self._poll_thread = threading.Thread(
            target=self._poll_telegram,
            name="FyersTelegramAuth",
            daemon=True,
        )
        self._poll_thread.start()
        LOGGER.info("Fyers Telegram authentication callback listening on port %s.", self.port)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the callback server and Telegram command poller."""
        self._stop_event.set()
        if self._http_thread.is_alive():
            self._httpd.shutdown()
        self._httpd.server_close()
        if self._http_thread.is_alive():
            self._http_thread.join(timeout=timeout)
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=timeout)

    def request_authorization(self, reason: str) -> None:
        """Send a fresh, one-use Fyers login link to the authorized chat."""
        state = secrets.token_urlsafe(32)
        with self._state_lock:
            self._pending_state = state
            self._pending_state_expires_at = time.monotonic() + AUTH_CODE_TTL_SECONDS
            self._auth_prompted = True
        auth_url = build_auth_url(self.client_id, self.redirect_uri, state)
        self._notify(
            f"{reason}\n\nOpen this Fyers authorization link and approve access:\n{auth_url}\n\n"
            "This one-use link expires in 10 minutes. The access token is never sent in Telegram."
        )

    def authenticate_startup(self, *, has_access_token: bool) -> None:
        """Validate the stored token or wait for a Telegram-approved token."""
        if has_access_token:
            try:
                self.market_data_client.validate_session()
                return
            except (FyersAPIError, RuntimeError):
                reason = "The Fyers access token is invalid or expired."
        else:
            reason = "First-time setup: approve Fyers access to start the trading service."

        LOGGER.warning("%s Waiting for Telegram-authorized Fyers access.", reason)
        self.request_authorization(reason)
        self._token_updated.wait()
        self.market_data_client.validate_session()
        LOGGER.info("Fyers access token approved and validated; continuing startup.")

    def _handle_callback(self, request_target: str) -> None:
        query = parse_qs(urlsplit(request_target).query, max_num_fields=8)
        state_values = query.get("state", [])
        code_values = query.get("auth_code", [])
        if not code_values or not code_values[0].strip():
            raise ValueError("Fyers callback did not include an authorization code.")
        if not state_values or not self._consume_state(state_values[0]):
            raise ValueError("Fyers callback state was missing, expired, or did not match.")
        if query.get("s", ["ok"])[0].lower() != "ok":
            raise ValueError("Fyers reported an unsuccessful authorization.")

        access_token = exchange_auth_code(
            self.client_id,
            self.secret_key,
            code_values[0].strip(),
        )

        def persist(token: str) -> None:
            persist_access_token(self.token_file, token)

        self.market_data_client.replace_access_token(access_token, persist=persist)
        self.market_data_client.validate_session()
        self._token_updated.set()
        self._auth_prompted = False
        self._notify(
            "Fyers authentication succeeded. The token is active in the running service "
            "and saved to persistent storage."
        )
        LOGGER.info("Fyers access token refreshed and persisted; token value was not logged.")

    def _consume_state(self, returned_state: str) -> bool:
        with self._state_lock:
            expected = self._pending_state
            expires_at = self._pending_state_expires_at
            if (
                expected is None
                or time.monotonic() > expires_at
                or not secrets.compare_digest(returned_state, expected)
            ):
                return False
            self._pending_state = None
            self._pending_state_expires_at = 0.0
            return True

    def _poll_telegram(self) -> None:
        while not self._stop_event.is_set():
            try:
                response = requests.get(
                    f"{TELEGRAM_API}/bot{self.bot_token}/getUpdates",
                    params={"offset": self._offset, "timeout": 20, "allowed_updates": '["message"]'},
                    timeout=HTTP_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or payload.get("ok") is not True:
                    raise RuntimeError("Telegram returned an invalid updates response.")
                updates = payload.get("result", [])
                if not isinstance(updates, list):
                    raise RuntimeError("Telegram returned an invalid updates list.")
                if not self._updates_initialized:
                    for update in updates:
                        update_id = update.get("update_id") if isinstance(update, dict) else None
                        if isinstance(update_id, int):
                            self._offset = max(self._offset, update_id + 1)
                    self._updates_initialized = True
                    continue
                for update in updates:
                    if not isinstance(update, dict):
                        continue
                    update_id = update.get("update_id")
                    if isinstance(update_id, int):
                        self._offset = max(self._offset, update_id + 1)
                    self._handle_update(update)
                self._check_session()
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                LOGGER.warning("Fyers Telegram command polling failed (%s).", type(exc).__name__)
                self._stop_event.wait(5.0)

    def _handle_update(self, update: dict[str, object]) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return
        chat = message.get("chat")
        if not isinstance(chat, dict) or chat.get("type") != "private":
            return
        chat_id = str(chat.get("id", ""))
        if not secrets.compare_digest(chat_id, self.authorized_chat_id):
            return
        text = message.get("text")
        if not isinstance(text, str):
            return
        command = text.strip().split(maxsplit=1)[0].split("@", maxsplit=1)[0].lower()
        if command in {"/auth", "/start"}:
            self.request_authorization("Fyers login requested. Complete the approval in your browser.")

    def _send_message(self, text: str) -> None:
        response = requests.post(
            f"{TELEGRAM_API}/bot{self.bot_token}/sendMessage",
            json={
                "chat_id": self.authorized_chat_id,
                "text": text,
                "disable_web_page_preview": True,
            },
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise RuntimeError("Telegram did not accept the authentication message.")

    def _notify(self, text: str) -> None:
        try:
            self._send_message(text)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            LOGGER.warning("Fyers Telegram notification failed (%s).", type(exc).__name__)

    def _check_session(self) -> None:
        if time.monotonic() < self._next_session_check_at:
            return
        self._next_session_check_at = time.monotonic() + SESSION_CHECK_INTERVAL_SECONDS
        try:
            self.market_data_client.validate_session()
        except RuntimeError:
            if not self._auth_prompted:
                self.request_authorization(
                    "Fyers rejected the current access token. Complete a new approval to restore market data."
                )
