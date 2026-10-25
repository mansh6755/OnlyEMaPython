"""Telegram-triggered Fyers OAuth with a guarded HTTPS callback endpoint."""

from __future__ import annotations

import json
import logging
import os
import secrets
import stat
import tempfile
import threading
import time
from datetime import date, datetime, time as day_time, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse, urlsplit
from zoneinfo import ZoneInfo

import requests

from Dependencies.fyers_api import FyersAPI, FyersAPIError, build_auth_url, exchange_auth_code
from Dependencies.fyers_dashboard_backtest import DashboardBacktestJobs

LOGGER = logging.getLogger(__name__)
TELEGRAM_API = "https://api.telegram.org"
AUTH_CODE_TTL_SECONDS = 10 * 60
SESSION_CHECK_INTERVAL_SECONDS = 5 * 60
AUTHENTICATION_TIME = day_time(9, 0)
IST = ZoneInfo("Asia/Kolkata")
BACKTEST_BLOCK_START = day_time(9, 15)
BACKTEST_BLOCK_END = day_time(15, 20)
ENV_FILE = Path(__file__).resolve().parent / ".env"
HTTP_TIMEOUT_SECONDS = (5.0, 30.0)
DASHBOARD_ASSETS_DIR = Path(__file__).resolve().parent / "dashboard_assets"
DASHBOARD_ASSET_ROUTES = {
    "/": ("fyers_dashboard.html", "text/html; charset=utf-8"),
    "/dashboard.css": ("fyers_dashboard.css", "text/css; charset=utf-8"),
    "/dashboard.js": ("fyers_dashboard.js", "text/javascript; charset=utf-8"),
    "/backtest": ("fyers_backtest.html", "text/html; charset=utf-8"),
    "/backtest.css": ("fyers_backtest.css", "text/css; charset=utf-8"),
    "/backtest.js": ("fyers_backtest.js", "text/javascript; charset=utf-8"),
    "/vendor/lightweight-charts.standalone.production.js": (
        "vendor/lightweight-charts.standalone.production.js",
        "text/javascript; charset=utf-8",
    ),
}


def next_daily_authentication_at(now: datetime) -> datetime:
    """Return the next 9:00 AM IST slot that falls Monday through Friday."""
    local_now = now.astimezone(IST)
    candidate_date = local_now.date()
    for _ in range(7):
        candidate = datetime.combine(candidate_date, AUTHENTICATION_TIME, tzinfo=IST)
        if candidate > local_now and candidate.weekday() < 5:
            return candidate
        candidate_date += timedelta(days=1)
    raise RuntimeError("Unable to find the next weekday Fyers authentication time.")


def dashboard_backtest_allowed_at(now: datetime) -> bool:
    local_now = now.astimezone(IST)
    return not (
        local_now.weekday() < 5
        and BACKTEST_BLOCK_START <= local_now.time() < BACKTEST_BLOCK_END
    )


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


def clear_saved_access_tokens(token_file: Path, env_file: Path = ENV_FILE) -> None:
    """Remove saved token copies before requesting a fresh Fyers authorization."""
    env_path = env_file.resolve()
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines(keepends=True)
        retained_lines = []
        for line in lines:
            assignment = line.lstrip()
            if assignment.startswith("export "):
                assignment = assignment[len("export ") :].lstrip()
            key, separator, _value = assignment.partition("=")
            if separator and key.strip() == "FYERS_ACCESS_TOKEN":
                continue
            retained_lines.append(line)

        if len(retained_lines) != len(lines):
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=env_file.parent,
                    prefix=f".{env_file.name}.",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    temporary.write("".join(retained_lines))
                    temporary.flush()
                    os.fsync(temporary.fileno())
                temporary_path.chmod(stat.S_IMODE(env_file.stat().st_mode))
                os.replace(temporary_path, env_file)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    if token_file.resolve() != env_path:
        token_file.unlink(missing_ok=True)


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
        dashboard_state: Any | None = None,
        dashboard_enabled: bool = False,
        dashboard_backtest_runner: Callable[[date, date], dict[str, Any]] | None = None,
        dashboard_last_trading_day: Callable[[], date] | None = None,
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
        if dashboard_enabled and dashboard_state is None:
            raise ValueError("Dashboard state is required when the dashboard is enabled.")

        self.bot_token = bot_token.strip()
        self.authorized_chat_id = authorized_chat_id.strip()
        self.client_id = client_id.strip()
        self.secret_key = secret_key.strip()
        self.redirect_uri = redirect_uri
        self.callback_path = callback_path
        self.token_file = token_file
        self.market_data_client = market_data_client
        self.port = port
        self.dashboard_enabled = dashboard_enabled
        self.dashboard_state = dashboard_state
        self.dashboard_backtest_jobs = (
            DashboardBacktestJobs(dashboard_backtest_runner)
            if dashboard_enabled and dashboard_backtest_runner
            else None
        )
        self.dashboard_last_trading_day = dashboard_last_trading_day
        self._dashboard_assets = (
            {
                route: (path.read_bytes(), content_type)
                for route, (relative_path, content_type) in DASHBOARD_ASSET_ROUTES.items()
                for path in (DASHBOARD_ASSETS_DIR / relative_path,)
            }
            if dashboard_enabled
            else {}
        )
        self._stop_event = threading.Event()
        self._token_updated = threading.Event()
        self._authorization_pending = threading.Event()
        self._state_lock = threading.Lock()
        self._pending_state: str | None = None
        self._pending_state_expires_at = 0.0
        self._offset = 0
        self._updates_initialized = False
        self._auth_prompted = False
        self._next_session_check_at = time.monotonic() + SESSION_CHECK_INTERVAL_SECONDS
        self._poll_thread: threading.Thread | None = None
        self._authentication_thread = threading.Thread(
            target=self._schedule_daily_authentication,
            name="FyersDailyAuthentication",
            daemon=True,
        )
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

            def _respond_bytes(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'",
                )
                self.end_headers()
                self.wfile.write(body)

            def _dashboard(self, request_path: str) -> bool:
                is_job_status = request_path.startswith("/api/backtest/")
                if request_path == "/api/backtest/default-date":
                    if not service.dashboard_enabled or service.dashboard_last_trading_day is None:
                        self._respond(404, "Backtest date lookup is disabled.")
                        return True
                    try:
                        trading_day = service.dashboard_last_trading_day()
                    except Exception as exc:
                        LOGGER.exception("Could not identify the latest Fyers trading day.")
                        self._respond(503, f"Unable to find the latest trading day ({type(exc).__name__}).")
                        return True
                    if not isinstance(trading_day, date):
                        self._respond(503, "Trading-day lookup returned an invalid date.")
                        return True
                    body = json.dumps(
                        {"last_trading_day": trading_day.isoformat()},
                        separators=(",", ":"),
                    ).encode("utf-8")
                    self._respond_bytes(200, body, "application/json; charset=utf-8")
                    return True
                if (
                    request_path not in DASHBOARD_ASSET_ROUTES
                    and request_path != "/api/dashboard"
                    and not is_job_status
                ):
                    return False
                if not service.dashboard_enabled:
                    self._respond(404, "Not found.")
                    return True
                if request_path == "/api/dashboard":
                    body = json.dumps(
                        service.dashboard_state.snapshot(),
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    self._respond_bytes(200, body, "application/json; charset=utf-8")
                    return True
                if is_job_status:
                    if not service.dashboard_backtest_jobs:
                        self._respond(404, "Backtests are disabled.")
                        return True
                    job_id = request_path.removeprefix("/api/backtest/")
                    job = service.dashboard_backtest_jobs.get(job_id)
                    if job is None:
                        self._respond(404, "Backtest job not found.")
                        return True
                    body = json.dumps(job, allow_nan=False, separators=(",", ":"), default=str).encode()
                    self._respond_bytes(200, body, "application/json; charset=utf-8")
                    return True
                body, content_type = service._dashboard_assets[request_path]
                self._respond_bytes(200, body, content_type)
                return True

            def _start_backtest(self) -> None:
                if not service.dashboard_enabled or service.dashboard_backtest_jobs is None:
                    self._respond(404, "Dashboard backtests are disabled.")
                    return
                if not dashboard_backtest_allowed_at(datetime.now(IST)):
                    self._respond(
                        409,
                        "Dashboard backtests are available outside the weekday market session only.",
                    )
                    return
                if self.headers.get_content_type() != "application/json":
                    self._respond(415, "Send a JSON request.")
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    if not 1 <= content_length <= 4096:
                        raise ValueError
                    payload = json.loads(self.rfile.read(content_length))
                    if not isinstance(payload, dict) or set(payload) != {
                        "start_date",
                        "end_date",
                    }:
                        raise ValueError
                    start_text = payload["start_date"]
                    end_text = payload["end_date"]
                    if not isinstance(start_text, str) or not isinstance(end_text, str):
                        raise ValueError
                    start = date.fromisoformat(start_text)
                    end = date.fromisoformat(end_text)
                    if start.isoformat() != start_text or end.isoformat() != end_text:
                        raise ValueError
                    today_ist = datetime.now(ZoneInfo("Asia/Kolkata")).date()
                    if start > end or end > today_ist:
                        raise ValueError
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    self._respond(400, "Provide valid start_date and end_date; end_date cannot be future.")
                    return
                job, accepted = service.dashboard_backtest_jobs.submit(start, end)
                body = json.dumps(job, allow_nan=False, separators=(",", ":"), default=str).encode()
                self._respond_bytes(
                    202 if accepted else 409,
                    body,
                    "application/json; charset=utf-8",
                )

            def do_GET(self) -> None:
                request_path = urlsplit(self.path).path.rstrip("/") or "/"
                if self._dashboard(request_path):
                    return
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
                request_path = urlsplit(self.path).path.rstrip("/") or "/"
                if request_path == "/api/backtest":
                    self._start_backtest()
                else:
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
        self._authentication_thread.start()
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
        if self._authentication_thread.is_alive():
            self._authentication_thread.join(timeout=timeout)

    def request_authorization(self, reason: str) -> None:
        """Send a fresh, one-use Fyers login link to the authorized chat."""
        self._authorization_pending.set()
        self._token_updated.clear()
        try:
            clear_saved_access_tokens(self.token_file, ENV_FILE)
        except (OSError, UnicodeError) as exc:
            LOGGER.error("Could not clear saved Fyers tokens; authorization link was not sent.")
            raise RuntimeError("Could not clear saved Fyers tokens.") from exc
        self.market_data_client.clear_access_token()
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

    def _schedule_daily_authentication(self) -> None:
        while not self._stop_event.is_set():
            scheduled_at = next_daily_authentication_at(datetime.now(IST))
            delay = max(0.0, (scheduled_at - datetime.now(IST)).total_seconds())
            if self._stop_event.wait(delay):
                return
            if self._stop_event.is_set():
                return
            try:
                self.request_authorization(
                    "Daily Fyers authentication: approve access to refresh today's trading session."
                )
            except RuntimeError as exc:
                LOGGER.error("Daily Fyers authorization prompt was not sent: %s", exc)

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
        self._authorization_pending.clear()
        self._token_updated.set()
        self._auth_prompted = False
        self._notify(
            "Fyers authentication succeeded. The token is active in the running service "
            "and saved to persistent storage."
        )
        LOGGER.info("Fyers access token refreshed and persisted; token value was not logged.")

    def wait_for_access_token(self) -> None:
        """Block trading startup while a requested Fyers authorization is pending."""
        if not self._authorization_pending.is_set():
            return
        LOGGER.info("Waiting for the requested Fyers authorization callback.")
        self._token_updated.wait()
        self.market_data_client.validate_session()

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
