"""Trend following strategies."""

from .moving_average_crossover import Strategy as MACrossStrategy
from .breakout import Strategy as BreakoutStrategy
from .momentum import Strategy as MomentumStrategy
from .adx_trend import Strategy as ADXStrategy

__all__ = ["MACrossStrategy", "BreakoutStrategy", "MomentumStrategy", "ADXStrategy"]
