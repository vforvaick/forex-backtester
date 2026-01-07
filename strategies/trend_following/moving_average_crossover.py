"""
Moving Average Crossover Strategy

Classic trend-following strategy using fast/slow moving average crossover.
"""

from typing import Any, Dict
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """Moving Average Crossover Strategy."""
    
    fast_period: int = 10
    slow_period: int = 50
    signal_type: str = "ema"  # ema or sma
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """
        Run backtest on data.
        
        Args:
            data: DataFrame with columns [timestamp, bid, ask, mid]
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate moving averages
        if self.signal_type == "ema":
            data = data.with_columns([
                pl.col("mid").ewm_mean(span=self.fast_period).alias("fast_ma"),
                pl.col("mid").ewm_mean(span=self.slow_period).alias("slow_ma"),
            ])
        else:  # sma
            data = data.with_columns([
                pl.col("mid").rolling_mean(window_size=self.fast_period).alias("fast_ma"),
                pl.col("mid").rolling_mean(window_size=self.slow_period).alias("slow_ma"),
            ])
        
        # Generate signals
        signals = self._generate_signals(data)
        
        # Calculate returns
        metrics = self._calculate_metrics(data, signals)
        metrics["params"] = {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_type": self.signal_type
        }
        
        return metrics
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate trading signals from crossover."""
        # Calculate moving averages if not present
        if "fast_ma" not in data.columns:
            if self.signal_type == "ema":
                data = data.with_columns([
                    pl.col("mid").ewm_mean(span=self.fast_period).alias("fast_ma"),
                    pl.col("mid").ewm_mean(span=self.slow_period).alias("slow_ma"),
                ])
            else:  # sma
                data = data.with_columns([
                    pl.col("mid").rolling_mean(window_size=self.fast_period).alias("fast_ma"),
                    pl.col("mid").rolling_mean(window_size=self.slow_period).alias("slow_ma"),
                ])

        # Signal: 1 when fast > slow, -1 when fast < slow
        return (
            pl.when(pl.col("fast_ma") > pl.col("slow_ma"))
            .then(1)
            .when(pl.col("fast_ma") < pl.col("slow_ma"))
            .then(-1)
            .otherwise(0)
        ).alias("signal")
    
    def _calculate_metrics(self, data: pl.DataFrame, signals: pl.Series) -> Dict[str, float]:
        """Calculate performance metrics with transaction costs."""
        from strategies.base import calculate_metrics_with_costs
        
        # Evaluate signal expression to Series if needed
        if isinstance(signals, pl.Expr):
            signals = data.select(signals).to_series()
        
        # Use centralized cost-aware metrics calculation
        metrics = calculate_metrics_with_costs(data, signals, symbol="EURUSD")
        
        # Add strategy params
        metrics["params"] = {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_type": self.signal_type
        }
        
        return metrics


# For module-level access
def create_strategy(**params) -> Strategy:
    """Factory function to create strategy with params."""
    return Strategy(**params)

