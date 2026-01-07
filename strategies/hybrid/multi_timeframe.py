"""
Multi-Timeframe Strategy

Uses higher timeframe for trend direction, lower for entry.
"""

from typing import Dict
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """Multi-timeframe trend confirmation."""
    
    tf_trend: str = "H4"  # Higher timeframe for trend
    tf_entry: str = "M15"  # Lower timeframe for entry
    trend_indicator: str = "ema_50"
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest."""
        # Simulate multi-timeframe by using different MA periods
        data = self._add_mtf_indicators(data)
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _add_mtf_indicators(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add multi-timeframe indicators."""
        # Parse trend indicator
        period = int(self.trend_indicator.split("_")[1]) if "_" in self.trend_indicator else 50
        
        # Higher TF trend (longer period)
        htf_mult = {"H4": 16, "D1": 96, "H1": 4}.get(self.tf_trend, 16)
        ltf_mult = {"M15": 1, "M5": 0.33, "M30": 2, "H1": 4}.get(self.tf_entry, 1)
        
        return data.with_columns([
            pl.col("mid").ewm_mean(span=int(period * htf_mult)).alias("trend_ma"),
            pl.col("mid").ewm_mean(span=int(period * ltf_mult)).alias("entry_ma"),
        ]).with_columns([
            (pl.col("mid") > pl.col("trend_ma")).alias("uptrend"),
            (pl.col("mid") < pl.col("trend_ma")).alias("downtrend"),
        ])
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate MTF signals."""
        # Add MTF indicators if not present
        if "uptrend" not in data.columns:
            data = self._add_mtf_indicators(data)
            
        # Entry only in direction of higher TF trend
        return (
            pl.when(pl.col("uptrend") & (pl.col("mid") > pl.col("entry_ma")))
            .then(1)  # Long in uptrend with entry confirmation
            .when(pl.col("downtrend") & (pl.col("mid") < pl.col("entry_ma")))
            .then(-1)  # Short in downtrend with entry confirmation
            .otherwise(0)
        ).alias("signal")
    
    def _calculate_metrics(self, data: pl.DataFrame, signals: pl.Series) -> Dict[str, float]:
        """Calculate metrics with transaction costs."""
        from strategies.base import calculate_metrics_with_costs
        
        if isinstance(signals, pl.Expr):
            signals = data.select(signals).to_series()
        
        return calculate_metrics_with_costs(data, signals, symbol="EURUSD")

