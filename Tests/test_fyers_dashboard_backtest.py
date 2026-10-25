from datetime import date
from threading import Event

import pandas as pd

from Dependencies.fyers_dashboard_backtest import DashboardBacktestJobs, summarize_backtest


def test_backtest_summary_contains_trade_stats_and_json_ready_records():
    trades = pd.DataFrame(
        [
            {
                "entry_time": "2026-10-20T10:00:00+05:30",
                "exit_time": "2026-10-20T10:10:00+05:30",
                "gross_pnl": -100.0,
            },
            {
                "entry_time": "2026-10-21T10:00:00+05:30",
                "exit_time": "2026-10-21T10:10:00+05:30",
                "gross_pnl": 200.0,
            },
        ]
    )

    result = summarize_backtest(
        trades,
        {"daily_loss_halts": 1, "missing_option_data": 0},
        3,
        instrument="NIFTY",
        start=date(2026, 10, 20),
        end=date(2026, 10, 21),
        daily_loss_cap=3500,
    )

    assert result["summary"] == {
        "index_signals": 3,
        "option_trades": 2,
        "wins": 1,
        "losses": 1,
        "win_rate_pct": 50.0,
        "gross_pnl": 100.0,
        "profit_factor": 2.0,
        "max_drawdown": -100.0,
        "daily_loss_cap": 3500,
        "daily_loss_halts": 1,
        "missing_option_data": 0,
    }
    assert result["trades"][0]["gross_pnl"] == -100.0


def test_dashboard_backtest_jobs_allow_only_one_running_job():
    started = Event()
    release = Event()

    def run(start, end):
        started.set()
        release.wait(timeout=2)
        return {"start": start.isoformat(), "end": end.isoformat()}

    jobs = DashboardBacktestJobs(run)
    start = date(2026, 10, 20)
    end = date(2026, 10, 21)
    first, accepted = jobs.submit(start, end)
    assert accepted is True
    assert started.wait(timeout=1)
    second, accepted = jobs.submit(start, end)
    assert accepted is False
    assert second["id"] == first["id"]

    release.set()
    for _ in range(100):
        completed = jobs.get(first["id"])
        if completed is not None and completed["status"] != "RUNNING":
            break
        Event().wait(0.01)
    assert completed["status"] == "COMPLETED"
    assert completed["result"] == {
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
