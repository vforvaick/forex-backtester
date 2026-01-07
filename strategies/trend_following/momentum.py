"""
Momentum Strategy

RSI, MACD, and Stochastic momentum indicators.
"""

from typing import Any, Dict
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """Momentum-based trading strategy."""
    
    # RSI params
    rsi_period: int = 14
    overbought: int = 70
    oversold: int = 30
    
    # MACD params (alternative)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Stochastic params (alternative)
    stoch_k: int = 14
    stoch_d: int = 3
    
    indicator: str = "rsi"  # rsi, macd, stochastic
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest on data."""
        if self.indicator == "macd":
            data = self._add_macd(data)
        elif self.indicator == "stochastic":
            data = self._add_stochastic(data)
        else:
            data = self._add_rsi(data)
        
        signals = self._generate_signals(data)
        metrics = self._calculate_metrics(data, signals)
        return metrics
    
    def _add_rsi(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add RSI indicator."""
        delta = pl.col("mid").diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0)
        
        return data.with_columns([
            gain.rolling_mean(window_size=self.rsi_period).alias("avg_gain"),
            loss.rolling_mean(window_size=self.rsi_period).alias("avg_loss"),
        ]).with_columns([
            (100 - 100 / (1 + pl.col("avg_gain") / pl.col("avg_loss"))).alias("rsi")
        ])
    
    def _add_macd(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add MACD indicator."""
        return data.with_columns([
            pl.col("mid").ewm_mean(span=self.macd_fast).alias("ema_fast"),
            pl.col("mid").ewm_mean(span=self.macd_slow).alias("ema_slow"),
        ]).with_columns([
            (pl.col("ema_fast") - pl.col("ema_slow")).alias("macd_line")
        ]).with_columns([
            pl.col("macd_line").ewm_mean(span=self.macd_signal).alias("signal_line")
        ])
    
    def _add_stochastic(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add Stochastic oscillator."""
        return data.with_columns([
            pl.col("mid").rolling_min(window_size=self.stoch_k).alias("low_k"),
            pl.col("mid").rolling_max(window_size=self.stoch_k).alias("high_k"),
        ]).with_columns([
            ((pl.col("mid") - pl.col("low_k")) / (pl.col("high_k") - pl.col("low_k")) * 100).alias("stoch_k")
        ]).with_columns([
            pl.col("stoch_k").rolling_mean(window_size=self.stoch_d).alias("stoch_d")
        ])
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate momentum signals."""
        # Calculate indicators if not present
        if self.indicator == "macd" and "macd_line" not in data.columns:
            data = self._add_macd(data)
        elif self.indicator == "stochastic" and "stoch_k" not in data.columns:
            data = self._add_stochastic(data)
        elif self.indicator == "rsi" and "rsi" not in data.columns:
            data = self._add_rsi(data)
            
        if self.indicator == "macd":
            return (
                pl.when((pl.col("macd_line") > pl.col("signal_line")) & 
                        (pl.col("macd_line").shift(1) <= pl.col("signal_line").shift(1)))
                .then(1)
                .when((pl.col("macd_line") < pl.col("signal_line")) & 
                      (pl.col("macd_line").shift(1) >= pl.col("signal_line").shift(1)))
                .then(-1)
                .otherwise(0)
            ).alias("signal")
        elif self.indicator == "stochastic":
            return (
                pl.when((pl.col("stoch_k") < self.oversold) & (pl.col("stoch_k") > pl.col("stoch_d")))
                .then(1)
                .when((pl.col("stoch_k") > self.overbought) & (pl.col("stoch_k") < pl.col("stoch_d")))
                .then(-1)
                .otherwise(0)
            ).alias("signal")
        else:  # RSI
            return (
                pl.when(pl.col("rsi") < self.oversold)
                .then(1)
                .when(pl.col("rsi") > self.overbought)
                .then(-1)
                .otherwise(0)
            ).alias("signal")
    
    def _calculate_metrics(self, data: pl.DataFrame, signals: pl.Series) -> Dict[str, float]:
        """Calculate performance metrics with transaction costs."""
        from strategies.base import calculate_metrics_with_costs
        
        if isinstance(signals, pl.Expr):
            signals = data.select(signals).to_series()
        
        return calculate_metrics_with_costs(data, signals, symbol="EURUSD")

