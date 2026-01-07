"""
ATR Trailing Stop Strategy

Volatility-based trailing stop using ATR.
"""

from typing import Dict
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """ATR-based trailing stop strategy."""
    
    atr_period: int = 14
    multiplier: float = 2.0
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest."""
        data = self._add_atr_stops(data)
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _add_atr_stops(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add ATR-based trailing stops."""
        return data.with_columns([
            (pl.col("ask") - pl.col("bid")).rolling_mean(window_size=self.atr_period).alias("atr"),
        ]).with_columns([
            (pl.col("mid") - self.multiplier * pl.col("atr")).alias("long_stop"),
            (pl.col("mid") + self.multiplier * pl.col("atr")).alias("short_stop"),
        ]).with_columns([
            pl.col("long_stop").cum_max().alias("trailing_long"),
            pl.col("short_stop").cum_min().alias("trailing_short"),
        ])
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate trend signals with ATR stops."""
        # Calculate ATR stops if not present
        if "trailing_long" not in data.columns:
            data = self._add_atr_stops(data)
            
        # Simple trend following: long when price above trailing stop
        return (
            pl.when(pl.col("mid") > pl.col("trailing_long"))
            .then(1)
            .when(pl.col("mid") < pl.col("trailing_short"))
            .then(-1)
            .otherwise(0)
        ).alias("signal")
    
    def _calculate_metrics(self, data: pl.DataFrame, signals: pl.Series) -> Dict[str, float]:
        """Calculate metrics with transaction costs."""
        from strategies.base import calculate_metrics_with_costs
        
        # Evaluate signal expression if needed
        if isinstance(signals, pl.Expr):
            signals = data.select(signals).to_series()
        
        return calculate_metrics_with_costs(data, signals, symbol="EURUSD")

