"""
Regime Switching Strategy

Switches between trend and range strategies based on market regime.
"""

from typing import Dict
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """Market regime detection and strategy switching."""
    
    adx_threshold: int = 25
    trend_strategy: str = "momentum"
    range_strategy: str = "mean_reversion"
    volatility_percentile: int = 50
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest."""
        data = self._detect_regime(data)
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _detect_regime(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect trending vs ranging market."""
        # Simplified ADX calculation
        price_change = pl.col("mid").diff().abs()
        
        return data.with_columns([
            price_change.rolling_mean(window_size=14).alias("directional_move"),
            (pl.col("ask") - pl.col("bid")).rolling_mean(window_size=14).alias("atr"),
        ]).with_columns([
            (pl.col("directional_move") / (pl.col("atr") + 0.0001) * 100).alias("adx_proxy"),
        ]).with_columns([
            (pl.col("adx_proxy") > self.adx_threshold).alias("trending"),
        ])
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate regime-adaptive signals."""
        # Detect regime if not present
        if "trending" not in data.columns:
            data = self._detect_regime(data)
            
        # Trend strategy: momentum
        trend_signal = (
            pl.when(pl.col("mid") > pl.col("mid").shift(10))
            .then(1)
            .when(pl.col("mid") < pl.col("mid").shift(10))
            .then(-1)
            .otherwise(0)
        )
        
        # Range strategy: mean reversion
        mid_mean = pl.col("mid").rolling_mean(window_size=20)
        range_signal = (
            pl.when(pl.col("mid") < mid_mean * 0.99)
            .then(1)
            .when(pl.col("mid") > mid_mean * 1.01)
            .then(-1)
            .otherwise(0)
        )
        
        # Switch based on regime
        return (
            pl.when(pl.col("trending"))
            .then(trend_signal)
            .otherwise(range_signal)
        ).alias("signal")
    
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
