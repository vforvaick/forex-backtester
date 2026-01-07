"""
Statistical Arbitrage Strategy

Z-score based mean reversion for price deviations.
"""

from typing import Dict
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """Z-score statistical arbitrage."""
    
    zscore_period: int = 20
    entry_zscore: float = 2.0
    exit_zscore: float = 0.0
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest."""
        data = self._add_zscore(data)
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _add_zscore(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add Z-score of price."""
        return data.with_columns([
            pl.col("mid").rolling_mean(window_size=self.zscore_period).alias("price_mean"),
            pl.col("mid").rolling_std(window_size=self.zscore_period).alias("price_std"),
        ]).with_columns([
            ((pl.col("mid") - pl.col("price_mean")) / (pl.col("price_std") + 0.0001)).alias("zscore")
        ])
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate z-score signals."""
        # Calculate Z-score if not present
        if "zscore" not in data.columns:
            data = self._add_zscore(data)
            
        return (
            pl.when(pl.col("zscore") < -self.entry_zscore)
            .then(1)  # Price too low, expect reversion up
            .when(pl.col("zscore") > self.entry_zscore)
            .then(-1)  # Price too high, expect reversion down
            .when(pl.col("zscore").abs() < self.exit_zscore + 0.5)
            .then(0)  # Exit near mean
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
