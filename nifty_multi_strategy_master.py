"""EMA-only trading runner for the trimmed repository.

This project was reduced to the EMA strategy only. The broad multi-strategy
master and all non-EMA worker implementations were removed, and this file keeps
only the import path and worker class needed to run the EMA logic.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
LOGGER_NAME = "ema_only_runner"
logger = logging.getLogger(LOGGER_NAME)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_module(module_name: str, file_path: Path) -> Any:
    """Load a Python module from a file path without requiring a package import."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing module file: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


EMA_LOGIC = load_module(
    "master_ema_trend_strategy_logic",
    ROOT_DIR / "Signal Generators" / "ema_trend_strategy_logic.py",
)

EMA_STRATEGY_CONFIG = EMA_LOGIC.EMATrendConfig(
    ema_fast_period=4,
    ema_mid_period=11,
    ema_slow_period=18,
    atr_period=14,
    adx_period=14,
    slope_lookback=3,
    adx_threshold=20.0,
    distance_atr_multiplier=0.5,
    ema11_slope_atr_multiplier=0.3,
    ema18_slope_atr_multiplier=0.2,
)

STRATEGY_ENV_PREFIX = {"EMA": "EMA"}
SIGNAL_GEN_WORKERS: list[type] = []
SL_HUNTING_AVAILABLE = False


class EMATrendStrategyWorker:
    """Minimal EMA strategy worker for the trimmed EMA-only repo."""

    strategy_name = "EMA"
    timeframe = "1"
    poll_seconds = 2
    derived_timeframe_minutes = 5
    lots = 1
    max_loss = 5500.0
    trading_start_hour = 9
    trading_start_minute = 25
    square_off_hour = 15
    square_off_minute = 15

    def __init__(self, store: Any | None = None, stop_event: Any | None = None, broker: Any | None = None):
        self.store = store
        self.stop_event = stop_event
        self.broker = broker
        self.signal_engine = EMA_LOGIC.EMATrendSignalEngine(EMA_STRATEGY_CONFIG)
        self.completed_trades = 0
        self.realized_pnl = 0.0
        self.pos = type("Pos", (), {"active": False, "direction": None})()

    def process_strategy_frame(self, strategy_frame):
        """Execute the EMA strategy on the provided OHLC frame."""
        return self.signal_engine.evaluate_candle(strategy_frame)


def main() -> None:
    logger.info("EMA-only trading runner active.")
    logger.info("Non-EMA strategy workers have been removed from this build.")


if __name__ == "__main__":
    main()
