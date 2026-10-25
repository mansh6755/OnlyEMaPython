"""Select exactly one NIFTY or SENSEX EMA9 forward runner from the environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

def selected_instrument(environment: dict[str, str]) -> str:
    instrument = environment.get("EMA9_INSTRUMENT", "").strip().upper()
    if instrument not in {"NIFTY", "SENSEX"}:
        raise ValueError("Set EMA9_INSTRUMENT to exactly one of NIFTY or SENSEX.")
    return instrument


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    load_dotenv(root_dir / ".env", override=False)
    instrument = selected_instrument(dict(os.environ))
    if instrument == "NIFTY":
        from Dependencies.fyers_nifty_ema9_forward_runner import main as run
    else:
        from Dependencies.fyers_sensex_ema9_forward_runner import main as run
    run()


if __name__ == "__main__":
    main()
