"""
Transaction Cost Model for Forex Backtesting

Provides realistic cost modeling including spread, commission, and slippage.
Critical for preventing overfitting and producing professional results.
"""

from typing import Dict
from dataclasses import dataclass
import polars as pl


@dataclass
class TransactionCostConfig:
    """
    Transaction cost configuration.
    
    Default values are typical for major forex pairs (EURUSD, GBPUSD).
    Adjust for exotic pairs or metals (XAUUSD uses higher spreads).
    """
    
    # Spread costs
    spread_pips: float = 0.5       # Half-spread (one-way) in pips
    spread_variable: bool = False  # If True, use actual bid-ask from data
    
    # Commission
    commission_per_lot: float = 7.0  # USD per round-trip (100k standard lot)
    
    # Slippage
    slippage_pips: float = 0.1     # Market impact / execution slippage
    
    # Position sizing
    lot_size: float = 100000       # Standard lot
    pip_value: float = 0.0001      # 4-digit pair (use 0.00001 for 5-digit)
    
    def total_cost_pips(self) -> float:
        """Total round-trip cost in pips (spread + slippage)."""
        return (self.spread_pips * 2) + self.slippage_pips
    
    @classmethod
    def for_eurusd(cls) -> 'TransactionCostConfig':
        """Standard config for EURUSD."""
        return cls(spread_pips=0.5, pip_value=0.0001)
    
    @classmethod
    def for_xauusd(cls) -> 'TransactionCostConfig':
        """Standard config for Gold (XAUUSD) - wider spreads."""
        return cls(spread_pips=2.5, pip_value=0.01, commission_per_lot=10.0)
    
    @classmethod
    def for_gbpusd(cls) -> 'TransactionCostConfig':
        """Standard config for Cable (GBPUSD)."""
        return cls(spread_pips=0.8, pip_value=0.0001)


def apply_transaction_costs(
    returns: pl.Series,
    signals: pl.Series,
    prices: pl.Series,
    config: TransactionCostConfig
) -> pl.Series:
    """
    Apply transaction costs to strategy returns.
    
    This function deducts spread, commission, and slippage costs
    from raw strategy returns on each trade entry/exit.
    
    Args:
        returns: Raw strategy returns (signal * price_change)
        signals: Trading signals (-1, 0, 1)
        prices: Mid prices for cost calculation
        config: Transaction cost configuration
        
    Returns:
        Net returns after costs (pl.Series)
    """
    # Detect trades: signal changes indicate entries/exits
    signal_changes = signals.diff().abs().fill_null(0)
    trade_entries = signal_changes > 0
    
    # Spread cost as percentage of price (round-trip = 2x spread)
    spread_cost = (config.spread_pips * 2) * config.pip_value / prices
    
    # Commission as percentage of position value
    # Commission is per lot, so convert to % of trade value
    commission_cost = config.commission_per_lot / (config.lot_size * prices)
    
    # Slippage as percentage
    slippage_cost = config.slippage_pips * config.pip_value / prices
    
    # Total cost per trade event
    total_cost = spread_cost + commission_cost + slippage_cost
    
    # Apply costs only when trade occurs (signal change)
    cost_to_apply = total_cost * trade_entries.cast(pl.Float64)
    
    # Net returns = gross returns - costs
    net_returns = returns - cost_to_apply
    
    return net_returns


def apply_transaction_costs_to_pnl(
    data: pl.DataFrame,
    signals: pl.Series,
    config: TransactionCostConfig
) -> pl.Series:
    """
    Calculate net PnL series with transaction costs.
    
    This is the main function to use when you have raw signals
    and price data, and want cumulative PnL after costs.
    
    Args:
        data: DataFrame with 'mid' column
        signals: Trading signals (-1, 0, 1)
        config: Transaction cost configuration
        
    Returns:
        Cumulative net PnL series
    """
    # Raw returns
    raw_returns = data["mid"].pct_change().fill_null(0)
    
    # Strategy returns = lagged signal * returns
    strategy_returns = signals.shift(1).fill_null(0) * raw_returns
    
    # Apply costs
    net_returns = apply_transaction_costs(
        strategy_returns, 
        signals, 
        data["mid"], 
        config
    )
    
    return net_returns.cum_sum()


def calculate_cost_breakdown(
    signals: pl.Series,
    prices: pl.Series,
    config: TransactionCostConfig
) -> Dict[str, float]:
    """
    Calculate detailed breakdown of transaction costs.
    
    Useful for analyzing how much costs are eating into profits.
    
    Args:
        signals: Trading signals
        prices: Price series
        config: Cost configuration
        
    Returns:
        Dictionary with cost breakdown:
            - total_trades: Number of round-trip trades
            - spread_cost_total: Total spread in price units
            - commission_total_usd: Total commission in USD
            - slippage_total: Total slippage cost
            - total_cost_pct: All costs as % of price
    """
    # Count trade entries (signal changes / 2 for round trips)
    signal_changes = signals.diff().abs().fill_null(0)
    n_entries = int(signal_changes.sum())
    n_round_trips = n_entries // 2
    
    avg_price = prices.mean()
    
    # Cost calculations
    spread_total = n_round_trips * (config.spread_pips * 2) * config.pip_value
    commission_total = n_round_trips * config.commission_per_lot
    slippage_total = n_round_trips * config.slippage_pips * config.pip_value
    
    # Total as percentage of average price
    total_price_cost = spread_total + slippage_total
    commission_as_pct = commission_total / (config.lot_size * avg_price) if avg_price > 0 else 0
    
    return {
        "total_trades": n_round_trips,
        "trade_entries": n_entries,
        "spread_cost_pips": config.spread_pips * 2 * n_round_trips,
        "spread_cost_total": spread_total,
        "commission_total_usd": commission_total,
        "slippage_total": slippage_total,
        "total_cost_pips": (config.spread_pips * 2 + config.slippage_pips) * n_round_trips,
        "average_cost_per_trade_pips": config.total_cost_pips(),
    }


# Convenience wrappers for common pairs
COST_CONFIGS = {
    "EURUSD": TransactionCostConfig.for_eurusd(),
    "GBPUSD": TransactionCostConfig.for_gbpusd(),
    "XAUUSD": TransactionCostConfig.for_xauusd(),
    "DEFAULT": TransactionCostConfig(),  # Default for unknown pairs
}


def get_cost_config(symbol: str) -> TransactionCostConfig:
    """Get appropriate cost config for a symbol."""
    symbol_upper = symbol.upper()
    return COST_CONFIGS.get(symbol_upper, COST_CONFIGS["DEFAULT"])
