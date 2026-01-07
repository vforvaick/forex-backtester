"""
Base Strategy Class

Provides shared metric calculation with transaction cost support.
All strategies should inherit from BaseStrategy or use calculate_metrics_with_costs().
"""

from typing import Dict, Optional
from dataclasses import dataclass
import polars as pl

from engine.transaction_costs import (
    TransactionCostConfig, 
    apply_transaction_costs,
    calculate_cost_breakdown,
    get_cost_config
)


@dataclass
class BaseStrategy:
    """
    Base class for all trading strategies.
    
    Provides standardized metric calculation with transaction cost support.
    Subclasses should implement:
        - _generate_signals(data) -> pl.Series
        - Any strategy-specific initialization
    """
    
    # Default cost config (can be overridden)
    cost_config: Optional[TransactionCostConfig] = None
    symbol: str = "EURUSD"
    
    def __post_init__(self):
        if self.cost_config is None:
            self.cost_config = get_cost_config(self.symbol)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest with transaction costs."""
        # Ensure mid price exists
        if "mid" not in data.columns:
            data = data.with_columns([
                ((pl.col("ask") + pl.col("bid")) / 2).alias("mid")
            ])
        
        # Generate signals (implemented by subclass)
        signals = self._generate_signals(data)
        
        # Calculate metrics with costs
        return self._calculate_metrics_with_costs(data, signals)
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate trading signals. Override in subclass."""
        raise NotImplementedError("Subclass must implement _generate_signals()")
    
    def _calculate_metrics_with_costs(
        self, 
        data: pl.DataFrame, 
        signals: pl.Series
    ) -> Dict[str, float]:
        """
        Calculate performance metrics with transaction costs.
        
        This is the standardized metrics calculation that includes:
        - Spread costs
        - Commission costs
        - Slippage
        """
        # Add signals to data
        data = data.with_columns(signals.alias("signal"))
        
        # Calculate raw returns
        raw_returns = data["mid"].pct_change().fill_null(0)
        
        # Strategy returns (using lagged signals to avoid lookahead)
        strategy_returns = data["signal"].shift(1).fill_null(0) * raw_returns
        
        # Apply transaction costs
        net_returns = apply_transaction_costs(
            strategy_returns,
            data["signal"],
            data["mid"],
            self.cost_config
        )
        
        # Calculate metrics on net returns
        return self._compute_metrics(net_returns, data["signal"], data["mid"])
    
    def _compute_metrics(
        self, 
        returns: pl.Series, 
        signals: pl.Series,
        prices: pl.Series
    ) -> Dict[str, float]:
        """Compute standard performance metrics."""
        returns = returns.drop_nulls()
        
        if len(returns) == 0 or returns.std() == 0:
            return self._empty_metrics()
        
        # Annualization factor (assuming tick/minute data)
        ann_factor = (252 * 24 * 60) ** 0.5
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        # Sharpe ratio
        sharpe = float(mean_ret / std_ret * ann_factor) if std_ret > 0 else 0.0
        
        # Sortino ratio (downside deviation only)
        downside = returns.filter(returns < 0)
        downside_std = downside.std() if len(downside) > 0 else std_ret
        sortino = float(mean_ret / downside_std * ann_factor) if downside_std > 0 else 0.0
        
        # Cumulative returns and drawdown
        cum_returns = (1 + returns).cum_prod()
        running_max = cum_returns.cum_max()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = abs(float(drawdown.min())) if len(drawdown) > 0 else 0.0
        
        # Win rate
        wins = returns.filter(returns > 0)
        win_rate = float(len(wins) / len(returns)) if len(returns) > 0 else 0.0
        
        # Profit factor
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        losses = returns.filter(returns < 0)
        gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0
        
        # Total return
        total_return = float(returns.sum())
        
        # Calmar ratio
        calmar = float(total_return / max_dd) if max_dd > 0 else 0.0
        
        # Trade count
        signal_changes = signals.diff().abs().fill_null(0)
        total_trades = int(signal_changes.sum() / 2)  # Round trips
        
        # Cost breakdown
        cost_info = calculate_cost_breakdown(signals, prices, self.cost_config)
        
        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "total_return": total_return,
            "calmar": calmar,
            # Cost metrics
            "spread_cost_pips": cost_info["spread_cost_pips"],
            "commission_usd": cost_info["commission_total_usd"],
            "cost_per_trade_pips": cost_info["average_cost_per_trade_pips"],
        }
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict."""
        return {
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0, "total_trades": 0,
            "total_return": 0.0, "calmar": 0.0,
            "spread_cost_pips": 0.0, "commission_usd": 0.0,
            "cost_per_trade_pips": 0.0,
        }


def calculate_metrics_with_costs(
    data: pl.DataFrame,
    signals: pl.Series,
    cost_config: Optional[TransactionCostConfig] = None,
    symbol: str = "EURUSD"
) -> Dict[str, float]:
    """
    Standalone function to calculate metrics with transaction costs.
    
    Use this when you can't inherit from BaseStrategy.
    
    Args:
        data: DataFrame with 'mid' column
        signals: Trading signals (-1, 0, 1)
        cost_config: Cost configuration (optional, uses symbol default)
        symbol: Trading pair for default cost lookup
        
    Returns:
        Dictionary of performance metrics
    """
    if cost_config is None:
        cost_config = get_cost_config(symbol)
    
    # Ensure mid exists
    if "mid" not in data.columns:
        data = data.with_columns([
            ((pl.col("ask") + pl.col("bid")) / 2).alias("mid")
        ])
    
    # Calculate raw returns
    raw_returns = data["mid"].pct_change().fill_null(0)
    
    # Strategy returns
    strategy_returns = signals.shift(1).fill_null(0) * raw_returns
    
    # Apply costs
    net_returns = apply_transaction_costs(
        strategy_returns,
        signals,
        data["mid"],
        cost_config
    )
    
    # Compute metrics
    returns = net_returns.drop_nulls()
    
    if len(returns) == 0 or returns.std() == 0:
        return {
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0, "total_trades": 0,
            "total_return": 0.0, "calmar": 0.0,
            "spread_cost_pips": 0.0, "commission_usd": 0.0,
            "cost_per_trade_pips": 0.0,
        }
    
    ann_factor = (252 * 24 * 60) ** 0.5
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    sharpe = float(mean_ret / std_ret * ann_factor) if std_ret > 0 else 0.0
    
    downside = returns.filter(returns < 0)
    downside_std = downside.std() if len(downside) > 0 else std_ret
    sortino = float(mean_ret / downside_std * ann_factor) if downside_std > 0 else 0.0
    
    cum_returns = (1 + returns).cum_prod()
    running_max = cum_returns.cum_max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = abs(float(drawdown.min())) if len(drawdown) > 0 else 0.0
    
    wins = returns.filter(returns > 0)
    win_rate = float(len(wins) / len(returns)) if len(returns) > 0 else 0.0
    
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    losses = returns.filter(returns < 0)
    gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0
    
    total_return = float(returns.sum())
    calmar = float(total_return / max_dd) if max_dd > 0 else 0.0
    
    signal_changes = signals.diff().abs().fill_null(0)
    total_trades = int(signal_changes.sum() / 2)
    
    cost_info = calculate_cost_breakdown(signals, data["mid"], cost_config)
    
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "total_return": total_return,
        "calmar": calmar,
        "spread_cost_pips": cost_info["spread_cost_pips"],
        "commission_usd": cost_info["commission_total_usd"],
        "cost_per_trade_pips": cost_info["average_cost_per_trade_pips"],
    }
