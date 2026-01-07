"""
Candlestick Pattern Strategy

Trades based on common candlestick patterns.
"""

from typing import Dict, List
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """Candlestick pattern recognition."""
    
    patterns: List[str] = None
    confirm_period: int = 3
    
    def __init__(self, **kwargs):
        self.patterns = kwargs.get("patterns", ["doji", "engulfing"])
        self.confirm_period = kwargs.get("confirm_period", 3)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest."""
        data = self._detect_patterns(data)
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _detect_patterns(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect candlestick patterns."""
        # Calculate mid price if not present
        if "mid" not in data.columns:
            data = data.with_columns([
                ((pl.col("ask") + pl.col("bid")) / 2).alias("mid")
            ])
        
        # Calculate OHLC from tick data (approximate)
        data = data.with_columns([
            pl.col("mid").shift(1).alias("open"),
            pl.col("mid").alias("close"),
            (pl.col("mid") - pl.col("mid").shift(1)).alias("body"),
            (pl.col("ask") - pl.col("bid")).alias("range"),
        ])
        
        # Doji: small body relative to range
        data = data.with_columns([
            (pl.col("body").abs() < pl.col("range") * 0.1).alias("doji"),
        ])
        
        # Engulfing: current body engulfs previous
        data = data.with_columns([
            ((pl.col("body") > 0) & (pl.col("body").shift(1) < 0) & 
             (pl.col("body").abs() > pl.col("body").shift(1).abs())).alias("bullish_engulf"),
            ((pl.col("body") < 0) & (pl.col("body").shift(1) > 0) & 
             (pl.col("body").abs() > pl.col("body").shift(1).abs())).alias("bearish_engulf"),
        ])
        
        return data
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate signals from patterns."""
        # Detect patterns if not present
        if "bullish_engulf" not in data.columns:
            data = self._detect_patterns(data)
        
        # Build signal expression (no lookahead - use confirmed patterns only)
        signal_expr = (
            pl.when(pl.col("bullish_engulf"))
            .then(1)
            .when(pl.col("bearish_engulf"))
            .then(-1)
            .when(pl.col("doji"))  # Doji alone is neutral, wait for confirmation
            .then(0)
            .otherwise(0)
        ).alias("signal")
        
        # Evaluate expression against data to get Series
        return data.select(signal_expr).to_series()
    
    def _calculate_metrics(self, data: pl.DataFrame, signals: pl.Series) -> Dict[str, float]:
        """Calculate metrics with transaction costs."""
        from strategies.base import calculate_metrics_with_costs
        return calculate_metrics_with_costs(data, signals, symbol="EURUSD")


