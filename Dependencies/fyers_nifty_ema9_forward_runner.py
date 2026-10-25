"""NIFTY EMA9 + 2-step ITM forward-test entry point."""

from __future__ import annotations

from Dependencies.fyers_ema9_forward_runner import InstrumentConfig, run_forward_runner

CONFIG = InstrumentConfig(
    name="NIFTY",
    underlying="NIFTY",
    index_symbol="NSE:NIFTY50-INDEX",
    exchange="NSE",
    expiry_policy="NEAREST",
)


def main() -> None:
    run_forward_runner(CONFIG)


if __name__ == "__main__":
    main()
