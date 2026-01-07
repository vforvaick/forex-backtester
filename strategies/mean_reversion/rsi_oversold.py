"""
RSI Oversold Strategy

Buy when RSI is oversold, sell when RSI returns to neutral.
"""

from typing import Dict
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """RSI oversold/overbought mean reversion."""
    
    period: int = 14
    entry: int = 30  # Buy when RSI below this
    exit: int = 50   # Exit when RSI reaches this
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest."""
        data = self._add_rsi(data)
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _add_rsi(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add RSI indicator."""
        delta = pl.col("mid").diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0)
        
        return data.with_columns([
            gain.rolling_mean(window_size=self.period).alias("avg_gain"),
            loss.rolling_mean(window_size=self.period).alias("avg_loss"),
        ]).with_columns([
            (100 - 100 / (1 + pl.col("avg_gain") / (pl.col("avg_loss") + 0.0001))).alias("rsi")
        ])
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate signals on RSI extremes."""
        # Calculate RSI if not present
        if "rsi" not in data.columns:
            data = self._add_rsi(data)
            
        overbought = 100 - self.entry  # Mirror of entry
        return (
            pl.when(pl.col("rsi") < self.entry)
            .then(1)  # Oversold = buy
            .when(pl.col("rsi") > overbought)
            .then(-1)  # Overbought = sell
            .when(pl.col("rsi").is_between(self.exit - 5, self.exit + 5))
            .then(0)  # Exit near middle
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
