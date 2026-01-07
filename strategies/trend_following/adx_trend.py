"""
ADX Trend Strategy

Uses Average Directional Index to filter for strong trends.
"""

from typing import Dict
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """ADX-based trend following strategy."""
    
    adx_period: int = 14
    threshold: int = 25
    di_period: int = 14
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest on data."""
        data = self._add_adx(data)
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _add_adx(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add ADX and directional indicators."""
        # Simplified ADX calculation
        # +DM = current_high - prev_high if positive, else 0
        # -DM = prev_low - current_low if positive, else 0
        
        return data.with_columns([
            pl.col("mid").diff().alias("price_change"),
            (pl.col("ask") - pl.col("bid")).rolling_mean(window_size=self.adx_period).alias("atr"),
        ]).with_columns([
            # Simplified +DI and -DI using price momentum
            pl.when(pl.col("price_change") > 0)
            .then(pl.col("price_change").rolling_mean(window_size=self.di_period))
            .otherwise(0).alias("plus_di"),
            
            pl.when(pl.col("price_change") < 0)
            .then((-pl.col("price_change")).rolling_mean(window_size=self.di_period))
            .otherwise(0).alias("minus_di"),
        ]).with_columns([
            # DX = |+DI - -DI| / (+DI + -DI) * 100
            ((pl.col("plus_di") - pl.col("minus_di")).abs() / 
             (pl.col("plus_di") + pl.col("minus_di") + 0.0001) * 100).alias("dx")
        ]).with_columns([
            # ADX = smoothed DX
            pl.col("dx").rolling_mean(window_size=self.adx_period).alias("adx")
        ])
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate signals based on ADX and DI crossover."""
        # Calculate ADX if not present
        if "adx" not in data.columns:
            data = self._add_adx(data)
            
        return (
            pl.when((pl.col("adx") > self.threshold) & (pl.col("plus_di") > pl.col("minus_di")))
            .then(1)  # Strong uptrend
            .when((pl.col("adx") > self.threshold) & (pl.col("plus_di") < pl.col("minus_di")))
            .then(-1)  # Strong downtrend
            .otherwise(0)  # No trend or weak trend
        ).alias("signal")
    
    def _calculate_metrics(self, data: pl.DataFrame, signals: pl.Series) -> Dict[str, float]:
        """Calculate metrics with transaction costs."""
        from strategies.base import calculate_metrics_with_costs
        
        if isinstance(signals, pl.Expr):
            signals = data.select(signals).to_series()
        
        return calculate_metrics_with_costs(data, signals, symbol="EURUSD")

