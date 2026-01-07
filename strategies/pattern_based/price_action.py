"""
Price Action Strategy

Trades pin bars and inside bars.
"""

from typing import Dict
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """Price action patterns (pin bars, inside bars)."""
    
    pin_bar: bool = True
    inside_bar: bool = True
    lookback: int = 5
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest."""
        data = self._detect_patterns(data)
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _detect_patterns(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect price action patterns."""
        # Calculate pseudo-OHLC
        data = data.with_columns([
            pl.col("mid").rolling_max(window_size=self.lookback).alias("high"),
            pl.col("mid").rolling_min(window_size=self.lookback).alias("low"),
            (pl.col("mid") - pl.col("mid").shift(1)).alias("body"),
            (pl.col("ask") - pl.col("bid")).alias("range"),
        ])
        
        # Pin bar: long wick, small body
        if self.pin_bar:
            data = data.with_columns([
                ((pl.col("high") - pl.col("mid")) > pl.col("body").abs() * 2).alias("bearish_pin"),
                ((pl.col("mid") - pl.col("low")) > pl.col("body").abs() * 2).alias("bullish_pin"),
            ])
        else:
            data = data.with_columns([
                pl.lit(False).alias("bearish_pin"),
                pl.lit(False).alias("bullish_pin"),
            ])
        
        # Inside bar: current range within previous
        if self.inside_bar:
            data = data.with_columns([
                ((pl.col("high") < pl.col("high").shift(1)) & 
                 (pl.col("low") > pl.col("low").shift(1))).alias("inside"),
            ])
        else:
            data = data.with_columns([pl.lit(False).alias("inside")])
        
        return data
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate signals."""
        # Detect patterns if not present
        if "bullish_pin" not in data.columns:
            data = self._detect_patterns(data)
            
        return (
            pl.when(pl.col("bullish_pin"))
            .then(1)
            .when(pl.col("bearish_pin"))
            .then(-1)
            .when(pl.col("inside") & (pl.col("body").shift(-1) > 0))
            .then(1)  # Inside bar breakout up
            .when(pl.col("inside") & (pl.col("body").shift(-1) < 0))
            .then(-1)  # Inside bar breakout down
            .otherwise(0)
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
