"""
Breakout Strategy

Trades breakouts from price channels (Donchian) with ATR-based stops.
"""

from typing import Any, Dict
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """Donchian Channel Breakout Strategy."""
    
    period: int = 20
    atr_multiplier: float = 2.0
    atr_period: int = 14
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest on data."""
        # Calculate Donchian Channel
        data = data.with_columns([
            pl.col("mid").rolling_max(window_size=self.period).alias("upper"),
            pl.col("mid").rolling_min(window_size=self.period).alias("lower"),
        ])
        
        # Calculate ATR for stops
        data = self._add_atr(data)
        
        # Generate signals
        signals = self._generate_signals(data)
        metrics = self._calculate_metrics(data, signals)
        metrics["params"] = {"period": self.period, "atr_multiplier": self.atr_multiplier}
        
        return metrics
    
    def _add_atr(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add ATR column."""
        # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        # Simplified: use bid-ask spread + price change
        return data.with_columns([
            (pl.col("ask") - pl.col("bid") + 
             pl.col("mid").diff().abs()).rolling_mean(window_size=self.atr_period).alias("atr")
        ])
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate breakout signals."""
        # Calculate Donchian Channel if not present
        if "upper" not in data.columns:
            data = data.with_columns([
                pl.col("mid").rolling_max(window_size=self.period).alias("upper"),
                pl.col("mid").rolling_min(window_size=self.period).alias("lower"),
            ])
            
        return (
            pl.when(pl.col("mid") > pl.col("upper").shift(1))
            .then(1)  # Long on upper breakout
            .when(pl.col("mid") < pl.col("lower").shift(1))
            .then(-1)  # Short on lower breakout
            .otherwise(0)
        ).alias("signal")
    
    def _calculate_metrics(self, data: pl.DataFrame, signals: pl.Series) -> Dict[str, float]:
        """Calculate performance metrics with transaction costs."""
        from strategies.base import calculate_metrics_with_costs
        
        if isinstance(signals, pl.Expr):
            signals = data.select(signals).to_series()
        
        return calculate_metrics_with_costs(data, signals, symbol="EURUSD")

