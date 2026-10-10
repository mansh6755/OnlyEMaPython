from threading import Event

import requests

from Dependencies.telegram_notifier import TelegramNotifier


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"ok": True}


def test_notifier_sends_queued_message_to_configured_chat():
    delivered = Event()

    class _Session:
        def post(self, url, *, json, timeout):
            assert url.endswith("/botsecret-token/sendMessage")
            assert json["chat_id"] == "12345"
            assert json["text"] == "EMA entry signal"
            assert timeout == (3.0, 5.0)
            delivered.set()
            return _Response()

    notifier = TelegramNotifier(
        "secret-token",
        "12345",
        enabled=True,
        session=_Session(),
    )
    notifier.notify("EMA entry signal")

    assert delivered.wait(timeout=1.0)
    notifier.close()


def test_notifier_logs_failure_without_exposing_bot_token(caplog):
    class _Session:
        def post(self, *_args, **_kwargs):
            raise requests.RequestException(
                "failed at https://api.telegram.org/botsecret-token/sendMessage"
            )

    notifier = TelegramNotifier(
        "secret-token",
        "12345",
        enabled=True,
        session=_Session(),
    )
    notifier.notify("EMA entry signal")
    notifier._messages.join()
    notifier.close()

    assert "Telegram notification failed (RequestException)" in caplog.text
    assert "secret-token" not in caplog.text


def test_notifier_stays_disabled_without_configuration():
    notifier = TelegramNotifier(enabled=True)

    notifier.notify("EMA entry signal")

    assert notifier._thread is None


def test_error_notifications_are_throttled_by_category():
    delivered = []

    class _Session:
        def post(self, _url, *, json, **_kwargs):
            delivered.append(json["text"])
            return _Response()

    notifier = TelegramNotifier(
        "secret-token",
        "12345",
        enabled=True,
        session=_Session(),
    )

    assert notifier.notify_error("history:FyersAPIError", "history request failed")
    assert not notifier.notify_error("history:FyersAPIError", "history request failed again")
    notifier._messages.join()
    notifier.close()

    assert delivered == ["history request failed"]
