from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd
import pytest

STRATEGY_PATH = (
    Path(__file__).resolve().parents[1]
    / "Signal Generators"
    / "ema_trend_strategy_logic.py"
)
SPEC = spec_from_file_location("ema_trend_strategy_logic_under_test", STRATEGY_PATH)
assert SPEC is not None and SPEC.loader is not None
STRATEGY = module_from_spec(SPEC)
sys.modules[SPEC.name] = STRATEGY
SPEC.loader.exec_module(STRATEGY)


@pytest.mark.parametrize(
    ("direction", "ema9", "ema11", "close"),
    [
        ("LONG", 100.0, 95.0, 99.0),
        ("SHORT", 100.0, 105.0, 101.0),
    ],
)
def test_open_position_exits_on_ema9_close_cross_without_ema11_cross(
    direction,
    ema9,
    ema11,
    close,
):
    candles = pd.DataFrame(
        {
            "high": [102.0] * 24,
            "low": [98.0] * 24,
            "close": [100.0] * 23 + [close],
            "ema4": [101.0] * 24,
            "ema9": [101.0] * 23 + [ema9],
            "ema11": [102.0] * 23 + [ema11],
            "ema18": [103.0] * 24,
            "atr": [1.0] * 24,
            "adx": [25.0] * 24,
            "full_body_candle": [True] * 24,
            "ema11_slope": [1.0] * 24,
            "ema18_slope": [1.0] * 24,
            "ema11_slope_strength": [1.0] * 24,
            "ema18_slope_strength": [1.0] * 24,
            "ema11_delta_current": [1.0] * 24,
            "ema11_delta_previous": [0.5] * 24,
            "long_setup": [False] * 24,
            "short_setup": [False] * 24,
        }
    )
    position = STRATEGY.EMATrendPositionContext(
        direction=direction,
        entry_underlying=100.0,
    )

    decision = STRATEGY.EMATrendSignalEngine().evaluate_candle(candles, position)

    assert decision.action == "EXIT"
    assert decision.exit_reason == "EMA9_EXIT"


@pytest.mark.parametrize(
    ("direction", "ema9", "ema11", "close"),
    [
        ("LONG", 100.0, 105.0, 104.0),
        ("SHORT", 100.0, 95.0, 96.0),
    ],
)
def test_open_position_can_exit_on_ema11_close_cross_without_ema9_cross(
    direction,
    ema9,
    ema11,
    close,
):
    candles = pd.DataFrame(
        {
            "high": [102.0] * 24,
            "low": [98.0] * 24,
            "close": [100.0] * 23 + [close],
            "ema4": [101.0] * 24,
            "ema9": [101.0] * 23 + [ema9],
            "ema11": [102.0] * 23 + [ema11],
            "ema18": [103.0] * 24,
            "atr": [1.0] * 24,
            "adx": [25.0] * 24,
            "full_body_candle": [True] * 24,
            "ema11_slope": [1.0] * 24,
            "ema18_slope": [1.0] * 24,
            "ema11_slope_strength": [1.0] * 24,
            "ema18_slope_strength": [1.0] * 24,
            "ema11_delta_current": [1.0] * 24,
            "ema11_delta_previous": [0.5] * 24,
            "long_setup": [False] * 24,
            "short_setup": [False] * 24,
        }
    )
    position = STRATEGY.EMATrendPositionContext(
        direction=direction,
        entry_underlying=100.0,
    )

    decision = STRATEGY.EMATrendSignalEngine(
        STRATEGY.EMATrendConfig(exit_ema_period=11)
    ).evaluate_candle(candles, position)

    assert decision.action == "EXIT"
    assert decision.exit_reason == "EMA11_EXIT"


@pytest.mark.parametrize(
    ("direction", "high", "low"),
    [
        ("LONG", 102.0, 99.0),
        ("SHORT", 101.0, 98.0),
    ],
)
def test_ema_wick_touch_without_close_cross_does_not_exit(direction, high, low):
    candles = pd.DataFrame(
        {
            "high": [102.0] * 23 + [high],
            "low": [98.0] * 23 + [low],
            "close": [100.0] * 24,
            "ema4": [101.0] * 24,
            "ema9": [101.0] * 23 + [100.0],
            "ema11": [102.0] * 24,
            "ema18": [103.0] * 24,
            "atr": [1.0] * 24,
            "adx": [25.0] * 24,
            "full_body_candle": [True] * 24,
            "ema11_slope": [1.0] * 24,
            "ema18_slope": [1.0] * 24,
            "ema11_slope_strength": [1.0] * 24,
            "ema18_slope_strength": [1.0] * 24,
            "ema11_delta_current": [1.0] * 24,
            "ema11_delta_previous": [0.5] * 24,
            "long_setup": [False] * 24,
            "short_setup": [False] * 24,
        }
    )

    decision = STRATEGY.EMATrendSignalEngine().evaluate_candle(
        candles,
        STRATEGY.EMATrendPositionContext(direction=direction, entry_underlying=100.0),
    )

    assert decision.action == "HOLD"
