"""
Ensemble Strategy

Combines multiple strategies with voting or weighted signals.
"""

from typing import Dict, List
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """Ensemble of multiple strategies."""
    
    strategies: List[str] = None
    vote_threshold: int = 2
    weight: List[float] = None
    
    def __init__(self, **kwargs):
        self.strategies = kwargs.get("strategies", ["ma_cross", "rsi", "breakout"])
        self.vote_threshold = kwargs.get("vote_threshold", 2)
        self.weight = kwargs.get("weight", None)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest."""
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _add_component_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add individual strategy signals."""
        # MA Cross signal
        data = data.with_columns([
            pl.col("mid").ewm_mean(span=10).alias("fast_ma"),
            pl.col("mid").ewm_mean(span=50).alias("slow_ma"),
        ]).with_columns([
            pl.when(pl.col("fast_ma") > pl.col("slow_ma")).then(1)
            .when(pl.col("fast_ma") < pl.col("slow_ma")).then(-1)
            .otherwise(0).alias("ma_signal"),
        ])
        
        # RSI signal
        delta = pl.col("mid").diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0)
        
        data = data.with_columns([
            gain.rolling_mean(window_size=14).alias("avg_gain"),
            loss.rolling_mean(window_size=14).alias("avg_loss"),
        ]).with_columns([
            (100 - 100 / (1 + pl.col("avg_gain") / (pl.col("avg_loss") + 0.0001))).alias("rsi")
        ]).with_columns([
            pl.when(pl.col("rsi") < 30).then(1)
            .when(pl.col("rsi") > 70).then(-1)
            .otherwise(0).alias("rsi_signal"),
        ])
        
        # Breakout signal
        data = data.with_columns([
            pl.col("mid").rolling_max(window_size=20).alias("upper"),
            pl.col("mid").rolling_min(window_size=20).alias("lower"),
        ]).with_columns([
            pl.when(pl.col("mid") > pl.col("upper").shift(1)).then(1)
            .when(pl.col("mid") < pl.col("lower").shift(1)).then(-1)
            .otherwise(0).alias("breakout_signal"),
        ])
        
        return data
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Combine signals using voting or weighting."""
        data = self._add_component_signals(data)
        if self.weight:
            # Weighted average
            w = self.weight + [0] * (3 - len(self.weight))  # Pad to 3
            combined = (
                pl.col("ma_signal") * w[0] +
                pl.col("rsi_signal") * w[1] +
                pl.col("breakout_signal") * w[2]
            )
            signal_expr = (
                pl.when(combined > 0.3).then(1)
                .when(combined < -0.3).then(-1)
                .otherwise(0)
            ).alias("signal")
        else:
            # Vote counting
            vote_sum = pl.col("ma_signal") + pl.col("rsi_signal") + pl.col("breakout_signal")
            signal_expr = (
                pl.when(vote_sum >= self.vote_threshold).then(1)
                .when(vote_sum <= -self.vote_threshold).then(-1)
                .otherwise(0)
            ).alias("signal")
        
        return data.select(signal_expr).to_series()
    
    def _calculate_metrics(self, data: pl.DataFrame, signals: pl.Series) -> Dict[str, float]:
        """Calculate metrics with transaction costs."""
        from strategies.base import calculate_metrics_with_costs
        return calculate_metrics_with_costs(data, signals, symbol="EURUSD")

