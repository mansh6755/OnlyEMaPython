"""Non-blocking Telegram message delivery for trading notifications."""

from __future__ import annotations

import logging
import os
from queue import Full, Queue
from threading import Lock, Thread
from time import monotonic

import requests

LOGGER = logging.getLogger(__name__)
TELEGRAM_API = "https://api.telegram.org"


class TelegramNotifier:
    """Send queued messages to one Telegram chat without blocking the caller."""

    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
        *,
        enabled: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.session = session or requests.Session()
        self._messages: Queue[str | None] = Queue(maxsize=100)
        self._thread: Thread | None = None
        self._error_lock = Lock()
        self._last_error_notification: dict[str, float] = {}

        if not enabled:
            return
        if not self.bot_token or not self.chat_id:
            LOGGER.warning("Telegram notifications enabled but bot token or chat ID is missing.")
            return

        self._thread = Thread(target=self._send_queued_messages, name="TelegramNotifier", daemon=True)
        self._thread.start()

    @classmethod
    def from_environment(cls) -> TelegramNotifier:
        return cls(
            os.getenv("TELEGRAM_BOT_TOKEN", ""),
            os.getenv("TELEGRAM_CHAT_ID", ""),
            enabled=os.getenv("TELEGRAM_ENABLED", "false").strip().lower() == "true",
        )

    def notify(self, message: str) -> bool:
        if self._thread is None:
            return False
        try:
            self._messages.put_nowait(message)
        except Full:
            LOGGER.warning("Telegram notification queue is full; dropping a message.")
            return False
        return True

    def notify_error(
        self,
        key: str,
        message: str,
        *,
        cooldown_seconds: float = 300.0,
    ) -> bool:
        if self._thread is None:
            return False
        now = monotonic()
        with self._error_lock:
            last_sent = self._last_error_notification.get(key)
            if last_sent is not None and now - last_sent < cooldown_seconds:
                return False
            self._last_error_notification[key] = now
        if self.notify(message):
            return True
        with self._error_lock:
            self._last_error_notification.pop(key, None)
        return False

    def close(self, timeout: float = 3.0) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        try:
            self._messages.put(None, timeout=timeout)
        except Full:
            LOGGER.warning("Telegram notification queue did not drain before shutdown.")
            return
        self._thread.join(timeout=timeout)

    def _send_queued_messages(self) -> None:
        while True:
            message = self._messages.get()
            try:
                if message is None:
                    return
                response = self.session.post(
                    f"{TELEGRAM_API}/bot{self.bot_token}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": message,
                        "disable_web_page_preview": True,
                    },
                    timeout=(3.0, 5.0),
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or payload.get("ok") is not True:
                    raise RuntimeError("Telegram rejected the message.")
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                LOGGER.warning("Telegram notification failed (%s).", type(exc).__name__)
            finally:
                self._messages.task_done()
