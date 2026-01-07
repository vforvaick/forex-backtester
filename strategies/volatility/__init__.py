"""Volatility strategies."""

from .atr_trailing import Strategy as ATRTrailingStrategy
from .volatility_breakout import Strategy as VolBreakoutStrategy
from .range_trading import Strategy as RangeTradingStrategy

__all__ = ["ATRTrailingStrategy", "VolBreakoutStrategy", "RangeTradingStrategy"]
