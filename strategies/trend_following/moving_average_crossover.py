"""
Moving Average Crossover Strategy

Classic trend-following strategy using fast/slow moving average crossover.
"""

from typing import Any, Dict
from dataclasses import dataclass

import polars as pl


@dataclass
class Strategy:
    """Moving Average Crossover Strategy."""
    
    fast_period: int = 10
    slow_period: int = 50
    signal_type: str = "ema"  # ema or sma
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """
        Run backtest on data.
        
        Args:
            data: DataFrame with columns [timestamp, bid, ask, mid]
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate moving averages
        if self.signal_type == "ema":
            data = data.with_columns([
                pl.col("mid").ewm_mean(span=self.fast_period).alias("fast_ma"),
                pl.col("mid").ewm_mean(span=self.slow_period).alias("slow_ma"),
            ])
        else:  # sma
            data = data.with_columns([
                pl.col("mid").rolling_mean(window_size=self.fast_period).alias("fast_ma"),
                pl.col("mid").rolling_mean(window_size=self.slow_period).alias("slow_ma"),
            ])
        
        # Generate signals
        signals = self._generate_signals(data)
        
        # Calculate returns
        metrics = self._calculate_metrics(data, signals)
        metrics["params"] = {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_type": self.signal_type
        }
        
        return metrics
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate trading signals from crossover."""
        # Calculate moving averages if not present
        if "fast_ma" not in data.columns:
            if self.signal_type == "ema":
                data = data.with_columns([
                    pl.col("mid").ewm_mean(span=self.fast_period).alias("fast_ma"),
                    pl.col("mid").ewm_mean(span=self.slow_period).alias("slow_ma"),
                ])
            else:  # sma
                data = data.with_columns([
                    pl.col("mid").rolling_mean(window_size=self.fast_period).alias("fast_ma"),
                    pl.col("mid").rolling_mean(window_size=self.slow_period).alias("slow_ma"),
                ])

        # Signal: 1 when fast > slow, -1 when fast < slow
        return (
            pl.when(pl.col("fast_ma") > pl.col("slow_ma"))
            .then(1)
            .when(pl.col("fast_ma") < pl.col("slow_ma"))
            .then(-1)
            .otherwise(0)
        ).alias("signal")
    
    def _calculate_metrics(self, data: pl.DataFrame, signals: pl.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Add signals to data
        data = data.with_columns(signals)
        
        # Calculate returns
        returns = data.select([
            (pl.col("mid").pct_change() * pl.col("signal").shift(1)).alias("strategy_return")
        ])["strategy_return"]
        
        # Remove nulls
        returns = returns.drop_nulls()
        
        if len(returns) == 0:
            return {
                "sharpe": 0, "sortino": 0, "max_drawdown": 0,
                "win_rate": 0, "profit_factor": 0, "total_trades": 0,
                "total_return": 0, "calmar": 0
            }
        
        # Basic metrics
        total_return = returns.sum()
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        # Annualization factor (assuming minute data)
        ann_factor = (252 * 24 * 60) ** 0.5
        
        sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0
        
        # Sortino
        downside = returns.filter(returns < 0)
        downside_std = downside.std() if len(downside) > 0 else std_ret
        sortino = (mean_ret / downside_std * ann_factor) if downside_std > 0 else 0
        
        # Max drawdown
        cum_returns = (1 + returns).cum_prod()
        running_max = cum_returns.cum_max()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
        
        # Calmar
        calmar = (total_return / abs(max_dd)) if max_dd != 0 else 0
        
        # Win rate
        wins = returns.filter(returns > 0)
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        
        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        losses = returns.filter(returns < 0)
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Trade count (signal changes)
        signal_changes = data["signal"].diff().abs().sum()
        
        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": int(signal_changes / 2),  # Round trips
            "total_return": total_return,
            "calmar": calmar
        }


# For module-level access
def create_strategy(**params) -> Strategy:
    """Factory function to create strategy with params."""
    return Strategy(**params)
