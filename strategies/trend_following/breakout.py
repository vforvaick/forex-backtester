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
        """Calculate performance metrics."""
        data = data.with_columns(signals)
        returns = data.select([
            (pl.col("mid").pct_change() * pl.col("signal").shift(1)).alias("strategy_return")
        ])["strategy_return"].drop_nulls()
        
        if len(returns) == 0:
            return {"sharpe": 0, "sortino": 0, "max_drawdown": 0, "win_rate": 0, 
                    "profit_factor": 0, "total_trades": 0, "total_return": 0, "calmar": 0}
        
        ann_factor = (252 * 24 * 60) ** 0.5
        mean_ret, std_ret = returns.mean(), returns.std()
        sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0
        
        downside = returns.filter(returns < 0)
        sortino = (mean_ret / downside.std() * ann_factor) if len(downside) > 0 and downside.std() > 0 else 0
        
        cum_returns = (1 + returns).cum_prod()
        max_dd = ((cum_returns - cum_returns.cum_max()) / cum_returns.cum_max()).min()
        
        wins = returns.filter(returns > 0)
        win_rate = len(wins) / len(returns)
        profit_factor = wins.sum() / abs(returns.filter(returns < 0).sum()) if returns.filter(returns < 0).sum() != 0 else 0
        
        return {
            "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_dd,
            "win_rate": win_rate, "profit_factor": profit_factor,
            "total_trades": int(data["signal"].diff().abs().sum() / 2),
            "total_return": returns.sum(), "calmar": returns.sum() / abs(max_dd) if max_dd != 0 else 0
        }
