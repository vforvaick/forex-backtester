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
        """Calculate metrics."""
        data = data.with_columns(signals.alias("signal"))
        
        # Calculate returns based on signal
        data = data.with_columns([
            (pl.col("mid").pct_change() * pl.col("signal").shift(1)).alias("r"),
            (pl.col("signal") != pl.col("signal").shift(1)).alias("trade_trigger")
        ])
        
        returns = data["r"].drop_nulls()
        
        if len(returns) == 0 or returns.std() == 0:
            return {"sharpe": 0, "sortino": 0, "max_drawdown": 0, "win_rate": 0,
                    "profit_factor": 0, "total_trades": 0, "total_return": 0, "calmar": 0}
        
        # Count actual trades (signal changes from 0 to non-zero or sign change)
        trade_signals = data.filter(pl.col("trade_trigger") & (pl.col("signal") != 0))
        total_trades = len(trade_signals)
        
        # Calculate metrics
        ann = (252 * 24 * 60) ** 0.5  # Annualization for tick data
        sharpe = float(returns.mean() / returns.std() * ann)
        
        # Sortino (downside deviation)
        downside = returns.filter(returns < 0)
        sortino = float(returns.mean() / downside.std() * ann) if len(downside) > 0 and downside.std() > 0 else 0
        
        # Win rate
        wins = returns.filter(returns > 0)
        win_rate = float(len(wins) / len(returns)) if len(returns) > 0 else 0
        
        # Profit factor
        gross_profit = float(returns.filter(returns > 0).sum())
        gross_loss = abs(float(returns.filter(returns < 0).sum()))
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0
        
        # Max drawdown
        cumulative = returns.cum_sum()
        running_max = cumulative.cum_max()
        drawdown = running_max - cumulative
        max_drawdown = float(drawdown.max()) if len(drawdown) > 0 else 0
        
        # Total return
        total_return = float(returns.sum())
        
        # Calmar
        calmar = float(total_return / max_drawdown) if max_drawdown > 0 else 0
        
        return {
            "sharpe": sharpe, 
            "sortino": sortino, 
            "max_drawdown": max_drawdown, 
            "win_rate": win_rate,
            "profit_factor": profit_factor, 
            "total_trades": total_trades, 
            "total_return": total_return, 
            "calmar": calmar
        }

