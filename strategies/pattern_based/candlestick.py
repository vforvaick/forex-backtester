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
        # Calculate OHLC from tick data (approximate)
        data = data.with_columns([
            pl.col("mid").alias("open").shift(1),
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
            
        # Combine pattern signals
        return (
            pl.when(pl.col("bullish_engulf") | (pl.col("doji") & (pl.col("body").shift(-1) > 0)))
            .then(1)
            .when(pl.col("bearish_engulf") | (pl.col("doji") & (pl.col("body").shift(-1) < 0)))
            .then(-1)
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
