"""Run and retain one dashboard-triggered EMA9 backtest at a time."""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import date
from typing import Any, Callable

import pandas as pd

from Dependencies.secret_redaction import environment_secrets, redact_text

LOGGER = logging.getLogger(__name__)


def summarize_backtest(
    trades: pd.DataFrame,
    counters: dict[str, int],
    index_trade_count: int,
    *,
    instrument: str,
    start: date,
    end: date,
    daily_loss_cap: float,
) -> dict[str, Any]:
    pnl = pd.to_numeric(
        trades.get("gross_pnl", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax().clip(lower=0)
    factor = float(winners.sum() / abs(losers.sum())) if not losers.empty else None
    rows = [
        {
            key: None if pd.isna(value) else value.item() if hasattr(value, "item") else value
            for key, value in row.items()
        }
        for row in trades.to_dict(orient="records")
    ]
    return {
        "instrument": instrument,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "summary": {
            "index_signals": index_trade_count,
            "option_trades": len(trades),
            "wins": len(winners),
            "losses": len(losers),
            "win_rate_pct": float(len(winners) / len(pnl) * 100.0) if len(pnl) else 0.0,
            "gross_pnl": float(pnl.sum()),
            "profit_factor": factor,
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
            "daily_loss_cap": daily_loss_cap,
            "daily_loss_halts": counters["daily_loss_halts"],
            "missing_option_data": counters["missing_option_data"],
        },
        "trades": rows,
    }


class DashboardBacktestJobs:
    """A single-job worker; HTTP requests only enqueue and read snapshots."""

    def __init__(
        self,
        runner: Callable[[date, date], dict[str, Any]],
    ) -> None:
        self._runner = runner
        self._lock = threading.Lock()
        self._job: dict[str, Any] | None = None

    def submit(self, start: date, end: date) -> tuple[dict[str, Any], bool]:
        with self._lock:
            if self._job is not None and self._job["status"] == "RUNNING":
                return dict(self._job), False
            job_id = uuid.uuid4().hex
            self._job = {
                "id": job_id,
                "status": "RUNNING",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "result": None,
                "error": None,
            }
        threading.Thread(
            target=self._run,
            args=(job_id, start, end),
            name=f"DashboardBacktest-{job_id[:8]}",
            daemon=True,
        ).start()
        with self._lock:
            return dict(self._job), True

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            if self._job is None or self._job["id"] != job_id:
                return None
            return dict(self._job)

    def _run(self, job_id: str, start: date, end: date) -> None:
        try:
            result = self._runner(start, end)
        except Exception as exc:
            result = None
            details = redact_text(str(exc), environment_secrets(os.environ))
            error = f"{type(exc).__name__}: {details[:1000]}"
            LOGGER.error("Dashboard EMA9 backtest failed (%s): %s", type(exc).__name__, details[:1000])
            status = "FAILED"
        else:
            error = None
            status = "COMPLETED"
        with self._lock:
            if self._job is None or self._job["id"] != job_id:
                return
            self._job.update(status=status, result=result, error=error)
