"""
Bollinger Bands Strategy

Mean reversion using Bollinger Bands for overbought/oversold signals.
"""

from typing import Dict
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """Bollinger Bands mean reversion strategy."""
    
    period: int = 20
    std_dev: float = 2.0
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest on data."""
        data = self._add_bollinger(data)
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _add_bollinger(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add Bollinger Bands."""
        data = data.with_columns([
            pl.col("mid").rolling_mean(window_size=self.period).alias("bb_middle"),
            pl.col("mid").rolling_std(window_size=self.period).alias("bb_std"),
        ])
        
        data = data.with_columns([
            (pl.col("bb_middle") + self.std_dev * pl.col("bb_std")).alias("bb_upper"),
            (pl.col("bb_middle") - self.std_dev * pl.col("bb_std")).alias("bb_lower"),
        ])
        
        data = data.with_columns([
            ((pl.col("mid") - pl.col("bb_lower")) / 
             (pl.col("bb_upper") - pl.col("bb_lower") + 0.0001)).alias("bb_pct"),
        ])
        
        return data
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate mean reversion signals."""
        # Calculate Bollinger Bands if not present
        if "bb_upper" not in data.columns:
            data = self._add_bollinger(data)
            
        signal_expr = (
            pl.when(pl.col("mid") < pl.col("bb_lower"))
            .then(1)  # Buy when touching lower band
            .when(pl.col("mid") > pl.col("bb_upper"))
            .then(-1)  # Sell when touching upper band
            .when(pl.col("bb_pct").is_between(0.4, 0.6))
            .then(0)  # Exit at middle
            .otherwise(pl.lit(None))
        ).forward_fill().fill_null(0).alias("signal")
        
        return data.select(signal_expr).to_series()
    
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
