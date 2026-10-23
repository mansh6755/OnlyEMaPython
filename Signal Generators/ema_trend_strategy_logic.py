"""
Shared EMA trend strategy logic driven by normal OHLC candles.

This file is the "brain" of the EMA strategy.
Its job is to do two separate things:
1. Build all indicator columns the strategy needs.
2. Read the latest candle and decide whether the strategy wants to:
   - ENTER_LONG
   - ENTER_SHORT
   - EXIT
   - HOLD

Quick-start:
1. Call `build_ema_trend_with_indicators()` on OHLC candles.
2. Keep one `EMATrendSignalEngine()` instance alive across evaluations.
3. Pass the enriched DataFrame to `evaluate_candle(...)`.
4. If a trade is open, also pass `EMATrendPositionContext(...)`.
5. Use the returned decision to enter, exit, or hold.

Strategy summary from the brief:
1. Long trigger needs close above EMA 4, EMA 11, and EMA 18.
2. Short trigger needs close below EMA 4, EMA 11, and EMA 18.
3. Entries also require EMA ordering, ATR distance, EMA slope thresholds,
   slope-strength comparison, EMA11 slope expansion, ADX confirmation,
   and a full-bodied entry candle.
4. Exit is confirmed at candle close:
   - long exit when candle closes below the selected exit EMA
   - short exit when candle closes above the selected exit EMA

Beginner mental model:
- `build_ema_trend_with_indicators()` prepares all the numbers.
- `EMATrendSignalEngine.evaluate_candle()` reads those numbers.
- The rest of the codebase only has to ask: "What should I do now?"
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import talib


@dataclass(frozen=True)
class EMATrendConfig:
    """Configurable strategy parameters from the strategy brief."""

    ema_fast_period: int = 4
    ema_mid_period: int = 11
    ema_slow_period: int = 18
    atr_period: int = 14
    adx_period: int = 14
    slope_lookback: int = 3
    adx_threshold: float = 20.0
    distance_atr_multiplier: float = 0.5
    ema11_slope_atr_multiplier: float = 0.3
    ema18_slope_atr_multiplier: float = 0.2
    full_body_min_ratio: float = 0.5
    exit_ema_period: int = 9

    def __post_init__(self) -> None:
        invalid_finite = [
            name
            for name, value in vars(self).items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
        ]
        if invalid_finite:
            raise ValueError(
                "EMA strategy configuration values must be finite. Invalid: "
                + ", ".join(invalid_finite)
            )
        # A negative or zero period would make no mathematical sense for EMA,
        # ATR, ADX, or lookback-based slope calculations. We validate here once
        # so the rest of the file can assume the config is sane.
        numeric_periods = {
            "ema_fast_period": self.ema_fast_period,
            "ema_mid_period": self.ema_mid_period,
            "ema_slow_period": self.ema_slow_period,
            "atr_period": self.atr_period,
            "adx_period": self.adx_period,
            "slope_lookback": self.slope_lookback,
        }
        invalid = [name for name, value in numeric_periods.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"All period values must be positive. Invalid: {', '.join(invalid)}")
        if not (
            int(self.ema_fast_period)
            < int(self.ema_mid_period)
            < int(self.ema_slow_period)
        ):
            raise ValueError("Require ema_fast_period < ema_mid_period < ema_slow_period.")
        if int(self.exit_ema_period) not in {9, 11}:
            raise ValueError("`exit_ema_period` must be 9 or 11.")
        non_negative = {
            "adx_threshold": self.adx_threshold,
            "distance_atr_multiplier": self.distance_atr_multiplier,
            "ema11_slope_atr_multiplier": self.ema11_slope_atr_multiplier,
            "ema18_slope_atr_multiplier": self.ema18_slope_atr_multiplier,
        }
        invalid_non_negative = [
            name for name, value in non_negative.items() if float(value) < 0.0
        ]
        if invalid_non_negative:
            raise ValueError(
                "EMA thresholds and multipliers must be non-negative. Invalid: "
                + ", ".join(invalid_non_negative)
            )
        if not (0.0 <= float(self.full_body_min_ratio) <= 1.0):
            raise ValueError("`full_body_min_ratio` must be between 0.0 and 1.0.")


@dataclass
class EMATrendPositionContext:
    """
    Minimal snapshot of an already-open trade.

    `stop_underlying` is optional because this strategy exits on the current
    EMA9 value, not on a fixed static stop price.
    """

    direction: str
    entry_underlying: float
    stop_underlying: float = 0.0


@dataclass
class EMATrendDecision:
    """
    Standard response object returned by the signal engine.

    `action` can be:
    - HOLD
    - ENTER_LONG
    - ENTER_SHORT
    - EXIT
    """

    action: str = "HOLD"
    entry_underlying: float = 0.0
    stop_underlying: float = 0.0
    exit_reason: str = ""
    signal_triggered: bool = False


def _require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Fail early with a clear message when required OHLC columns are missing."""
    # This helper prevents mysterious downstream errors like KeyError or NaN
    # calculations by checking the input schema at the boundary of each function.
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _resolve_timestamp(ohlc: pd.DataFrame) -> pd.Series:
    """Accept either a `timestamp` column or a datetime-like index."""
    # Many data sources keep time in a real column, but some code uses the
    # DataFrame index for dates. This function accepts both so callers do not
    # have to normalize that detail themselves.
    if "timestamp" in ohlc.columns:
        timestamp = pd.to_datetime(ohlc["timestamp"], errors="coerce")
    else:
        timestamp = pd.to_datetime(pd.Series(ohlc.index, index=ohlc.index), errors="coerce")

    # If every value is invalid, the strategy cannot sort candles in historical
    # order and would risk evaluating signals incorrectly.
    if timestamp.isna().all():
        raise ValueError("A valid 'timestamp' column or datetime-like index is required.")
    return timestamp


def _build_long_setup(frame: pd.DataFrame, config: EMATrendConfig) -> pd.Series:
    """Combine every bullish filter from the strategy brief."""
    # This returns one boolean value per candle.
    # True means that candle satisfies every bullish entry rule at the same time.
    # The `full_body_candle` filter ensures the breakout candle is not a weak
    # indecisive bar with long wicks and a tiny body.
    atr = frame["atr"]
    return (
        (frame["close"] > frame["ema4"])
        & (frame["close"] > frame["ema11"])
        & (frame["close"] > frame["ema18"])
        & (frame["ema4"] > frame["ema11"])
        & (frame["ema11"] > frame["ema18"])
        & (frame["ema_distance"] > config.distance_atr_multiplier * atr)
        & (frame["ema18_slope"] > 0.0)
        & (frame["ema11_slope"] > 0.0)
        & (frame["ema18_slope"] >= config.ema18_slope_atr_multiplier * atr)
        & (frame["ema11_slope"] >= config.ema11_slope_atr_multiplier * atr)
        & (frame["ema11_slope_strength"] > frame["ema18_slope_strength"])
        & (frame["ema11_delta_current"] > frame["ema11_delta_previous"])
        & (frame["adx"] > config.adx_threshold)
        & frame["full_body_candle"]
    )


def _build_short_setup(frame: pd.DataFrame, config: EMATrendConfig) -> pd.Series:
    """Combine every bearish filter from the strategy brief."""
    # This is the exact mirror image of the long setup:
    # price below EMAs, bearish EMA order, negative slopes, strong trend, etc.
    # The same full-bodied-candle filter is also applied here so bearish
    # breakouts must also be driven by a strong real body.
    atr = frame["atr"]
    return (
        (frame["close"] < frame["ema4"])
        & (frame["close"] < frame["ema11"])
        & (frame["close"] < frame["ema18"])
        & (frame["ema4"] < frame["ema11"])
        & (frame["ema11"] < frame["ema18"])
        & (frame["ema_distance"] < -config.distance_atr_multiplier * atr)
        & (frame["ema18_slope"] < 0.0)
        & (frame["ema11_slope"] < 0.0)
        & (frame["ema18_slope"] <= -config.ema18_slope_atr_multiplier * atr)
        & (frame["ema11_slope"] <= -config.ema11_slope_atr_multiplier * atr)
        & (frame["ema11_slope_strength"] < frame["ema18_slope_strength"])
        & (frame["ema11_delta_current"] < frame["ema11_delta_previous"])
        & (frame["adx"] > config.adx_threshold)
        & frame["full_body_candle"]
    )


def build_ema_trend_with_indicators(
    ohlc: pd.DataFrame,
    config: EMATrendConfig | None = None,
) -> pd.DataFrame:
    """
    Enrich OHLC candles with the indicators and derived signals used by the strategy.

    Output columns added:
    - timestamp
    - ema4, ema9, ema11, ema18
    - atr
    - adx
    - candle_body, candle_range, candle_body_ratio, full_body_candle
    - ema_distance
    - ema11_slope, ema18_slope
    - ema11_slope_strength, ema18_slope_strength
    - ema11_delta_current, ema11_delta_previous
    - long_setup, short_setup
    - long_exit, short_exit
    """
    # If caller does not pass custom settings, fall back to the standard
    # strategy values defined in the config dataclass.
    config = config or EMATrendConfig()
    _require_columns(ohlc, ["open", "high", "low", "close"])

    # Work on a copy so this function never mutates the caller's original
    # DataFrame. That makes debugging much safer and prevents accidental side
    # effects in notebooks or live scripts.
    frame = ohlc.copy()
    frame["timestamp"] = _resolve_timestamp(frame)
    # The strategy must read candles in true time order. Duplicates are removed
    # so one candle is never counted twice.
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Convert numeric columns safely. Any bad strings become NaN instead of
    # crashing the strategy mid-run.
    numeric_columns = [column for column in ("open", "high", "low", "close") if column in frame.columns]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # If a candle lost its essential price fields during conversion, we drop it.
    # A half-broken candle is worse than no candle.
    frame = frame.dropna(subset=["high", "low", "close"]).reset_index(drop=True)

    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)

    # A single pinned backend keeps live and backtest warm-up/smoothing
    # semantics identical on every supported Python version.
    close_values = np.asarray(close, dtype=np.float64)
    high_values = np.asarray(high, dtype=np.float64)
    low_values = np.asarray(low, dtype=np.float64)
    frame["ema4"] = talib.EMA(close_values, timeperiod=config.ema_fast_period)
    frame["ema9"] = talib.EMA(close_values, timeperiod=9)
    frame["ema11"] = talib.EMA(close_values, timeperiod=config.ema_mid_period)
    frame["ema18"] = talib.EMA(close_values, timeperiod=config.ema_slow_period)
    frame["atr"] = talib.ATR(
        high_values,
        low_values,
        close_values,
        timeperiod=config.atr_period,
    )
    frame["adx"] = talib.ADX(
        high_values,
        low_values,
        close_values,
        timeperiod=config.adx_period,
    )

    # Candle-structure filter:
    # The user defined a "full-bodied" candle as:
    # abs(close - open) >= 90% * abs(high - low)
    #
    # We store both the raw body/range values and the ratio so debugging is easy.
    frame["candle_body"] = (frame["close"] - frame["open"]).abs()
    frame["candle_range"] = (frame["high"] - frame["low"]).abs()
    safe_range = frame["candle_range"].replace(0.0, np.nan)
    frame["candle_body_ratio"] = frame["candle_body"] / safe_range
    frame["full_body_candle"] = (
        (frame["candle_range"] > 0.0)
        & (frame["candle_body"] >= config.full_body_min_ratio * frame["candle_range"])
    )

    # ATR can theoretically be zero during flat data. Replacing zero with NaN
    # prevents divide-by-zero errors when we calculate slope strength.
    safe_atr = frame["atr"].replace(0.0, np.nan)
    # EMA distance tells us how far apart the fast and slow EMAs are. The brief
    # requires that this distance be large enough relative to ATR.
    frame["ema_distance"] = frame["ema4"] - frame["ema18"]
    # The slope is defined by the user as current EMA minus EMA from `n`
    # candles ago. Because `n` is configurable, we use the config value here.
    frame["ema11_slope"] = frame["ema11"] - frame["ema11"].shift(config.slope_lookback)
    frame["ema18_slope"] = frame["ema18"] - frame["ema18"].shift(config.slope_lookback)
    # Strength turns the raw slope into a volatility-adjusted value by dividing
    # by ATR. This lets the strategy compare momentum relative to recent range.
    frame["ema11_slope_strength"] = frame["ema11_slope"] / safe_atr
    frame["ema18_slope_strength"] = frame["ema18_slope"] / safe_atr
    # These one-candle EMA11 changes are used for the "slope expansion" rule.
    frame["ema11_delta_current"] = frame["ema11"] - frame["ema11"].shift(1)
    frame["ema11_delta_previous"] = frame["ema11"].shift(1) - frame["ema11"].shift(2)
    # These final boolean columns are the most useful debugging fields.
    # Instead of manually recomputing conditions, a caller can inspect these
    # directly to see whether the latest candle is a valid setup or exit.
    #
    # Exit is confirmed by the completed candle close against the configured EMA.
    frame["long_setup"] = _build_long_setup(frame, config)
    frame["short_setup"] = _build_short_setup(frame, config)
    exit_ema = frame[f"ema{config.exit_ema_period}"]
    frame["long_exit"] = frame["close"] < exit_ema
    frame["short_exit"] = frame["close"] > exit_ema
    return frame


class EMATrendSignalEngine:
    """
    Signal engine for the EMA 4 / 11 / 18 trend strategy.

    Think of this class as the decision-maker that sits on top of the indicator
    DataFrame built by `build_ema_trend_with_indicators()`.
    """

    def __init__(self, config: EMATrendConfig | None = None):
        # Keeping config on the engine means backtests, front-tests, and any
        # future live-trading script can all share the exact same rules.
        self.config = config or EMATrendConfig()

    @staticmethod
    def _hold(signal_triggered: bool = False) -> EMATrendDecision:
        # Centralizing the "do nothing" response keeps every HOLD branch
        # returning the same object shape.
        return EMATrendDecision(action="HOLD", signal_triggered=signal_triggered)

    @staticmethod
    def _normalize_direction(direction: str) -> str:
        # Users and frameworks often pass values like "long", "LONG ", etc.
        # Normalizing once avoids repeated string-cleaning bugs.
        return str(direction).strip().upper()

    def _minimum_history_bars(self) -> int:
        # The strategy needs enough candles for:
        # 1. EMA warm-up,
        # 2. ATR/ADX warm-up,
        # 3. the configurable slope lookback,
        # 4. and the 2 extra candles needed for slope expansion.
        return max(
            self.config.ema_fast_period,
            self.config.ema_mid_period,
            self.config.ema_slow_period,
            self.config.atr_period,
            self.config.adx_period,
        ) + self.config.slope_lookback + 2

    def _evaluate_exit(
        self,
        position: EMATrendPositionContext,
        current_candle: pd.Series,
    ) -> EMATrendDecision:
        """
        Exit an open trade when a completed candle closes across the configured EMA.

        The caller evaluates this only after the candle has completed, so wick
        touches without a close across the EMA do not exit the position.
        """
        direction = self._normalize_direction(position.direction)
        close = float(current_candle["close"])
        exit_ema_name = f"ema{self.config.exit_ema_period}"
        exit_ema = float(current_candle[exit_ema_name])
        exit_reason = f"EMA{self.config.exit_ema_period}_EXIT"

        if direction == "LONG" and close < exit_ema:
            return EMATrendDecision(action="EXIT", exit_reason=exit_reason)

        if direction == "SHORT" and close > exit_ema:
            return EMATrendDecision(action="EXIT", exit_reason=exit_reason)

        return self._hold()

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: EMATrendPositionContext | None = None,
    ) -> EMATrendDecision:
        """
        Evaluate the latest candle and return one decision.

        Entry rules:
        - Long: close above all EMAs plus all bullish filters from the brief,
          including the full-bodied-candle filter.
        - Short: close below all EMAs plus all bearish filters from the brief,
          including the full-bodied-candle filter.
        - Exit: completed candle closes across the configured exit EMA against
          the active trade direction.
        """
        required_columns = [
            "high",
            "low",
            "close",
            "ema4",
            "ema9",
            "ema11",
            "ema18",
            "atr",
            "adx",
            "full_body_candle",
            "ema11_slope",
            "ema18_slope",
            "ema11_slope_strength",
            "ema18_slope_strength",
            "ema11_delta_current",
            "ema11_delta_previous",
            "long_setup",
            "short_setup",
        ]
        if candles_with_indicators is None:
            return self._hold()
        _require_columns(candles_with_indicators, required_columns)

        # Warm-up guard: without enough history, EMAs/ATR/ADX/slope values are
        # not fully formed yet and the safest action is to do nothing.
        if len(candles_with_indicators) < self._minimum_history_bars():
            return self._hold()

        current = candles_with_indicators.iloc[-1]
        # Even after the DataFrame is long enough, a malformed row or partial
        # history slice could still contain NaN values. We refuse to trade on
        # incomplete information.
        needed_values = pd.Series(
            [
                current["high"],
                current["low"],
                current["close"],
                current["ema4"],
                current["ema9"],
                current["ema11"],
                current["ema18"],
                current["atr"],
                current["adx"],
                current["full_body_candle"],
                current["ema11_slope"],
                current["ema18_slope"],
                current["ema11_slope_strength"],
                current["ema18_slope_strength"],
                current["ema11_delta_current"],
                current["ema11_delta_previous"],
            ]
        )
        if needed_values.isna().any():
            return self._hold()

        # If a position is already open, the only question is whether the trade
        # should remain open or be exited.
        if position is not None:
            return self._evaluate_exit(position, current)

        close = float(current["close"])
        exit_ema = float(current[f"ema{self.config.exit_ema_period}"])

        # Flat-state branch:
        # If the latest candle satisfies the full bullish/bearish setup, the
        # engine returns an entry decision and includes the current EMA9 value
        # as a helpful reference stop for logging and downstream systems.
        if bool(current["long_setup"]):
            return EMATrendDecision(
                action="ENTER_LONG",
                entry_underlying=close,
                stop_underlying=exit_ema,
                signal_triggered=True,
            )

        if bool(current["short_setup"]):
            return EMATrendDecision(
                action="ENTER_SHORT",
                entry_underlying=close,
                stop_underlying=exit_ema,
                signal_triggered=True,
            )

        return self._hold()
