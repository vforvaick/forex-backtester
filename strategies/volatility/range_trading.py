"""
Range Trading Strategy

Trades within support/resistance range.
"""

from typing import Dict
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """Support/Resistance range trading."""
    
    lookback: int = 20
    breakout_pct: float = 0.5
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest."""
        data = self._add_range(data)
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _add_range(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate support/resistance levels."""
        return data.with_columns([
            pl.col("mid").rolling_max(window_size=self.lookback).alias("resistance"),
            pl.col("mid").rolling_min(window_size=self.lookback).alias("support"),
        ]).with_columns([
            (pl.col("resistance") - pl.col("support")).alias("range_size"),
            ((pl.col("mid") - pl.col("support")) / 
             (pl.col("resistance") - pl.col("support") + 0.0001)).alias("range_pct"),
        ])
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate range trading signals."""
        # Calculate range if not present
        if "range_pct" not in data.columns:
            data = self._add_range(data)
            
        return (
            pl.when(pl.col("range_pct") < 0.2)
            .then(1)  # Near support = buy
            .when(pl.col("range_pct") > 0.8)
            .then(-1)  # Near resistance = sell
            .when(pl.col("range_pct").is_between(0.4, 0.6))
            .then(0)  # Exit in middle
            .otherwise(pl.lit(None))
        ).forward_fill().fill_null(0).alias("signal")
    
    def _calculate_metrics(self, data: pl.DataFrame, signals: pl.Series) -> Dict[str, float]:
        """Calculate metrics."""
        data = data.with_columns(signals)
        returns = data.select([
            (pl.col("mid").pct_change() * pl.col("signal").shift(1)).alias("r")
        ])["r"].drop_nulls()
        
        if len(returns) == 0:
            return {"sharpe": 0, "sortino": 0, "max_drawdown": 0, "win_rate": 0,
                    "profit_factor": 0, "total_trades": 0, "total_return": 0, "calmar": 0}
        
        ann = (252 * 24 * 60) ** 0.5
        sharpe = (returns.mean() / returns.std() * ann) if returns.std() > 0 else 0
        
        return {"sharpe": sharpe, "sortino": 0, "max_drawdown": 0, "win_rate": 0,
                "profit_factor": 0, "total_trades": 0, "total_return": returns.sum(), "calmar": 0}
