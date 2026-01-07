"""
HftBacktest Engine Wrapper

Wraps HftBacktest for forex backtesting with proper configuration.
"""

from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass

import polars as pl

# Note: Install with: pip install hftbacktest
try:
    import hftbacktest as hbt
    from hftbacktest import LIMIT, BUY, SELL, GTX
    HFT_AVAILABLE = True
except ImportError:
    print("HftBacktest not installed. Run: pip install hftbacktest")
    HFT_AVAILABLE = False


@dataclass
class ForexConfig:
    """Configuration for forex backtesting."""
    symbol: str
    tick_size: float = 0.00001  # 1 pip for 5-digit pairs
    lot_size: float = 100000   # Standard lot
    
    # Transaction costs (CRITICAL for realistic metrics)
    spread_pips: float = 0.5   # Half-spread in pips (typical EURUSD)
    commission_per_lot: float = 7.0  # USD per round-trip lot
    slippage_pips: float = 0.1  # Execution slippage
    
    leverage: int = 100
    
    # Latency modeling
    feed_latency_ms: float = 1.0
    order_latency_ms: float = 5.0
    
    @property
    def total_cost_pips(self) -> float:
        """Total round-trip cost in pips."""
        return (self.spread_pips * 2) + self.slippage_pips


class Backtester:
    """
    Forex backtester using HftBacktest engine.
    
    Handles data loading, simulation, and metric calculation.
    """
    
    def __init__(self, config: ForexConfig):
        self.config = config
        self.results = None
        
    def load_data(self, data_path: Path, start: str, end: str) -> pl.DataFrame:
        """Load and filter tick data."""
        # Handle directory or specific file
        if data_path.is_dir():
            path = data_path / "*.parquet"
        else:
            path = data_path
            
        from datetime import datetime
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
            
        df = pl.scan_parquet(path).filter(
            (pl.col("timestamp") >= start_dt) &
            (pl.col("timestamp") <= end_dt)
        ).collect()
        
        return df
    
    def run(
        self,
        strategy_func: callable,
        data: pl.DataFrame,
        params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Run backtest with given strategy.
        
        Args:
            strategy_func: Strategy function that takes (backtester, params)
            data: Tick data DataFrame
            params: Strategy parameters
        
        Returns:
            Dictionary of performance metrics
        """
        if not HFT_AVAILABLE:
            return self._run_simple_backtest(strategy_func, data, params)
        
        # Convert data to HftBacktest format
        # This is a simplified example - actual implementation depends on
        # HftBacktest's data format requirements
        
        # TODO: Implement full HftBacktest integration
        # For now, use simple backtesting
        return self._run_simple_backtest(strategy_func, data, params)
    
    def _run_simple_backtest(
        self,
        strategy_func: callable,
        data: pl.DataFrame,
        params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Simple vectorized backtest fallback.
        
        Used when HftBacktest is not available or for quick testing.
        """
        # Calculate mid price and spread
        data = data.with_columns([
            ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
            (pl.col("ask") - pl.col("bid")).alias("spread")
        ])
        
        # Run strategy - now expects metrics directly
        result = strategy_func(data, params)
        
        # If strategy returns metrics dict, use it directly
        if isinstance(result, dict):
            return result
        
        # Otherwise, treat as signals and calculate metrics
        signals = result
        
        # Evaluate signals if they are expressions
        if isinstance(signals, pl.Expr):
            signals = data.select(signals).to_series()
        
        # Calculate PnL from signals
        pnl = self._calculate_pnl(data, signals)
        
        # Calculate metrics
        metrics = self._calculate_metrics(pnl)
        
        return metrics
    
    def _calculate_pnl(self, data: pl.DataFrame, signals: pl.Series) -> pl.Series:
        """Calculate PnL from signals."""
        # Simple implementation - actual would account for:
        # - Spread costs
        # - Slippage
        # - Commission
        # - Position sizing
        
        returns = data["mid"].pct_change()
        strategy_returns = signals.shift(1) * returns
        
        # Apply spread cost
        spread_cost = self.config.spread_pips * self.config.tick_size / data["mid"]
        trades = signals.diff().abs()
        strategy_returns = strategy_returns - (trades * spread_cost)
        
        return strategy_returns.cum_sum()
    
    def _calculate_metrics(self, pnl: pl.Series) -> Dict[str, float]:
        """Calculate performance metrics from PnL series."""
        returns = pnl.diff().drop_nulls()
        
        if len(returns) == 0:
            return {
                "sharpe": 0, "sortino": 0, "max_drawdown": 0,
                "win_rate": 0, "profit_factor": 0, "total_trades": 0,
                "total_return": 0
            }
        
        # Annualized metrics (assuming minute-level data)
        annual_factor = (252 * 24 * 60) ** 0.5
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        sharpe = (mean_return / std_return * annual_factor) if std_return > 0 else 0
        
        # Sortino (downside deviation)
        downside = returns.filter(returns < 0)
        downside_std = downside.std() if len(downside) > 0 else std_return
        sortino = (mean_return / downside_std * annual_factor) if downside_std > 0 else 0
        
        # Max drawdown
        cummax = pnl.cum_max()
        drawdown = (pnl - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Win rate
        wins = returns.filter(returns > 0)
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        
        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        losses = returns.filter(returns < 0)
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(returns.filter(returns != 0)),
            "total_return": pnl[-1] if len(pnl) > 0 else 0
        }
