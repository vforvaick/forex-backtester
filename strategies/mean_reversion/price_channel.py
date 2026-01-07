"""
Price Channel Strategy

Keltner Channel and Envelope-based mean reversion.
"""

from typing import Dict
from dataclasses import dataclass

import polars as pl


@dataclass  
class Strategy:
    """Price channel mean reversion."""
    
    period: int = 20
    channel_type: str = "keltner"  # keltner or envelope
    pct: float = 0.5  # For envelope: percentage distance
    atr_mult: float = 2.0  # For keltner: ATR multiplier
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest."""
        if self.channel_type == "keltner":
            data = self._add_keltner(data)
        else:
            data = self._add_envelope(data)
        
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _add_keltner(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add Keltner Channel."""
        return data.with_columns([
            pl.col("mid").ewm_mean(span=self.period).alias("kc_middle"),
            (pl.col("ask") - pl.col("bid")).rolling_mean(window_size=self.period).alias("atr"),
        ]).with_columns([
            (pl.col("kc_middle") + self.atr_mult * pl.col("atr")).alias("channel_upper"),
            (pl.col("kc_middle") - self.atr_mult * pl.col("atr")).alias("channel_lower"),
        ])
    
    def _add_envelope(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add Price Envelope."""
        return data.with_columns([
            pl.col("mid").rolling_mean(window_size=self.period).alias("env_middle"),
        ]).with_columns([
            (pl.col("env_middle") * (1 + self.pct / 100)).alias("channel_upper"),
            (pl.col("env_middle") * (1 - self.pct / 100)).alias("channel_lower"),
        ])
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate channel signals."""
        # Calculate indicators if not present
        if "channel_upper" not in data.columns:
            if self.channel_type == "keltner":
                data = self._add_keltner(data)
            else:
                data = self._add_envelope(data)
                
        return (
            pl.when(pl.col("mid") < pl.col("channel_lower"))
            .then(1)
            .when(pl.col("mid") > pl.col("channel_upper"))
            .then(-1)
            .otherwise(0)
        ).alias("signal")
    
    def _calculate_metrics(self, data: pl.DataFrame, signals: pl.Series) -> Dict[str, float]:
        """Calculate metrics with transaction costs."""
        from strategies.base import calculate_metrics_with_costs
        
        if isinstance(signals, pl.Expr):
            signals = data.select(signals).to_series()
        
        return calculate_metrics_with_costs(data, signals, symbol="EURUSD")

