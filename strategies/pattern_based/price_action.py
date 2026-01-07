"""
Price Action Strategy

Trades pin bars and inside bars with proper confirmation.
No lookahead bias - signals only on confirmed patterns.
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
    
    # Pin bar thresholds
    pin_body_ratio: float = 0.3  # Body must be < 30% of total range
    pin_wick_ratio: float = 2.0  # Rejecting wick must be 2x body size
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest."""
        # Calculate mid price if not present
        if "mid" not in data.columns:
            data = data.with_columns([
                ((pl.col("ask") + pl.col("bid")) / 2).alias("mid")
            ])
        
        data = self._detect_patterns(data)
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _detect_patterns(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect price action patterns."""
        # Calculate pseudo-OHLC from tick data
        data = data.with_columns([
            pl.col("mid").rolling_max(window_size=self.lookback).alias("high"),
            pl.col("mid").rolling_min(window_size=self.lookback).alias("low"),
            pl.col("mid").shift(1).alias("open"),
            pl.col("mid").alias("close"),
        ])
        
        # Calculate candle components
        data = data.with_columns([
            (pl.col("close") - pl.col("open")).alias("body"),
            (pl.col("high") - pl.col("low")).alias("range"),
            (pl.col("high") - pl.col("close").clip(lower_bound=pl.col("open"))).alias("upper_wick"),
            (pl.col("close").clip(upper_bound=pl.col("open")) - pl.col("low")).alias("lower_wick"),
        ])
        
        # Pin bar detection - long wick with small body indicating rejection
        if self.pin_bar:
            data = data.with_columns([
                # Bullish pin bar: long lower wick, small body, price rejected from lows
                ((pl.col("lower_wick") > pl.col("body").abs() * self.pin_wick_ratio) &
                 (pl.col("body").abs() < pl.col("range") * self.pin_body_ratio) &
                 (pl.col("range") > 0)).alias("bullish_pin"),
                
                # Bearish pin bar: long upper wick, small body, price rejected from highs
                ((pl.col("upper_wick") > pl.col("body").abs() * self.pin_wick_ratio) &
                 (pl.col("body").abs() < pl.col("range") * self.pin_body_ratio) &
                 (pl.col("range") > 0)).alias("bearish_pin"),
            ])
        else:
            data = data.with_columns([
                pl.lit(False).alias("bearish_pin"),
                pl.lit(False).alias("bullish_pin"),
            ])
        
        # Inside bar detection: current range fully within previous bar's range
        if self.inside_bar:
            data = data.with_columns([
                ((pl.col("high") < pl.col("high").shift(1)) & 
                 (pl.col("low") > pl.col("low").shift(1))).alias("inside"),
            ])
            
            # Inside bar breakout direction - use PREVIOUS bar's direction for signal
            # This avoids lookahead: we trade the breakout after it happens
            data = data.with_columns([
                # Confirmed breakout: was inside bar, now broke out
                (pl.col("inside").shift(1) & 
                 (pl.col("close") > pl.col("high").shift(1))).alias("inside_breakout_up"),
                (pl.col("inside").shift(1) & 
                 (pl.col("close") < pl.col("low").shift(1))).alias("inside_breakout_down"),
            ])
        else:
            data = data.with_columns([
                pl.lit(False).alias("inside"),
                pl.lit(False).alias("inside_breakout_up"),
                pl.lit(False).alias("inside_breakout_down"),
            ])
        
        return data
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate signals - no lookahead bias."""
        # Detect patterns if not present
        if "bullish_pin" not in data.columns:
            data = self._detect_patterns(data)
        
        # Build signal expression (priority: pin bars > inside bar breakouts)
        signal_expr = (
            pl.when(pl.col("bullish_pin"))
            .then(1)
            .when(pl.col("bearish_pin"))
            .then(-1)
            .when(pl.col("inside_breakout_up"))
            .then(1)
            .when(pl.col("inside_breakout_down"))
            .then(-1)
            .otherwise(0)
        ).alias("signal")
        
        return data.select(signal_expr).to_series()
    
    def _calculate_metrics(self, data: pl.DataFrame, signals: pl.Series) -> Dict[str, float]:
        """Calculate metrics with transaction costs."""
        from strategies.base import calculate_metrics_with_costs
        return calculate_metrics_with_costs(data, signals, symbol="EURUSD")

