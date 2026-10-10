from Dependencies.fyers_dashboard_state import FyersDashboardState


def test_dashboard_state_returns_isolated_snapshots_and_bounds_history():
    state = FyersDashboardState(max_candles=2, max_trades=2, max_events=2)
    state.update_market(candles=[{"time": 1}, {"time": 2}, {"time": 3}], spot=24000.0)
    state.record_trade({"symbol": "NIFTY-CE", "pnl": 50.0})
    state.record_trade({"symbol": "NIFTY-PE", "pnl": -10.0})
    state.record_trade({"symbol": "NIFTY-CE", "pnl": 20.0})
    state.record_event({"message": "started"})
    state.record_event({"message": "data fetched"})
    state.record_event({"message": "health"})

    snapshot = state.snapshot()
    snapshot["market"]["candles"].clear()
    snapshot["account"]["realized_pnl"] = 100000

    current = state.snapshot()
    assert current["market"]["spot"] == 24000.0
    assert current["market"]["candles"] == [{"time": 2}, {"time": 3}]
    assert [trade["pnl"] for trade in current["trades"]] == [-10.0, 20.0]
    assert [event["message"] for event in current["events"]] == ["data fetched", "health"]
    assert current["account"]["realized_pnl"] == 0.0


def test_dashboard_state_position_can_be_cleared():
    state = FyersDashboardState()
    state.set_position({"direction": "LONG", "symbol": "NIFTY-CE"})

    assert state.snapshot()["position"]["direction"] == "LONG"

    state.set_position(None)

    assert state.snapshot()["position"] is None
