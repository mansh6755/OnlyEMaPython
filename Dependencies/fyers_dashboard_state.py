"""Thread-safe, bounded snapshot state for the read-only Fyers EMA dashboard."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from threading import Lock
from typing import Any


class FyersDashboardState:
    def __init__(self, *, max_candles: int = 300, max_trades: int = 50, max_events: int = 50) -> None:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        self._lock = Lock()
        self._max_candles = max(1, int(max_candles))
        self._max_trades = max(1, int(max_trades))
        self._max_events = max(1, int(max_events))
        self._snapshot: dict[str, Any] = {
            "service": {
                "status": "STARTING",
                "mode": "PAPER",
                "strategy": "EMA",
                "started_at": now,
                "updated_at": now,
            },
            "market": {
                "status": "WAITING",
                "underlying": "NSE:NIFTY50-INDEX",
                "last_success_at": None,
                "bar_at": None,
                "spot": None,
                "ema_fast": None,
                "ema_mid": None,
                "ema_slow": None,
                "atr": None,
                "adx": None,
                "candles": [],
            },
            "account": {
                "realized_pnl": 0.0,
                "open_pnl": None,
                "total_pnl": 0.0,
                "trade_count": 0,
                "daily_loss_limit": None,
            },
            "position": None,
            "signal": {"action": "HOLD", "status": "WAITING", "at": None},
            "trades": [],
            "events": [],
        }

    def update_service(self, **values: Any) -> None:
        with self._lock:
            self._snapshot["service"].update(deepcopy(values))
            self._touch()

    def update_market(self, *, candles: list[dict[str, Any]] | None = None, **values: Any) -> None:
        with self._lock:
            market = self._snapshot["market"]
            market.update(deepcopy(values))
            if candles is not None:
                market["candles"] = deepcopy(candles[-self._max_candles :])
            self._touch()

    def update_account(self, **values: Any) -> None:
        with self._lock:
            self._snapshot["account"].update(deepcopy(values))
            self._touch()

    def update_signal(self, **values: Any) -> None:
        with self._lock:
            self._snapshot["signal"].update(deepcopy(values))
            self._touch()

    def set_position(self, position: dict[str, Any] | None) -> None:
        with self._lock:
            self._snapshot["position"] = deepcopy(position)
            self._touch()

    def record_trade(self, trade: dict[str, Any]) -> None:
        with self._lock:
            trades = self._snapshot["trades"]
            trades.append(deepcopy(trade))
            del trades[: max(0, len(trades) - self._max_trades)]
            self._touch()

    def record_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            events = self._snapshot["events"]
            events.append(deepcopy(event))
            del events[: max(0, len(events) - self._max_events)]
            self._touch()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._snapshot)

    def _touch(self) -> None:
        self._snapshot["service"]["updated_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
