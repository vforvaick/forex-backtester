"""
Volatility Breakout Strategy

Trades breakouts after volatility squeeze (BB squeeze).
"""

from typing import Dict
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """Volatility squeeze breakout."""
    
    bb_period: int = 20
    squeeze_threshold: float = 0.5
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest."""
        data = self._add_squeeze(data)
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _add_squeeze(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect volatility squeeze."""
        return data.with_columns([
            pl.col("mid").rolling_std(window_size=self.bb_period).alias("bb_width"),
        ]).with_columns([
            (pl.col("bb_width") / pl.col("bb_width").rolling_mean(window_size=self.bb_period * 2)).alias("squeeze_ratio"),
        ]).with_columns([
            (pl.col("squeeze_ratio") < self.squeeze_threshold).alias("in_squeeze"),
        ])
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate breakout signals after squeeze."""
        # Calculate squeeze if not present
        if "in_squeeze" not in data.columns:
            data = self._add_squeeze(data)
            
        # Breakout direction based on price movement after squeeze
        return (
            pl.when(
                pl.col("in_squeeze").shift(1) & 
                ~pl.col("in_squeeze") & 
                (pl.col("mid") > pl.col("mid").shift(1))
            ).then(1)  # Upward breakout
            .when(
                pl.col("in_squeeze").shift(1) & 
                ~pl.col("in_squeeze") & 
                (pl.col("mid") < pl.col("mid").shift(1))
            ).then(-1)  # Downward breakout
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
