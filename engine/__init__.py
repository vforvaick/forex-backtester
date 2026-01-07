"""Engine module for backtesting."""

from .backtester import Backtester, ForexConfig
from .parallel_runner import (
    run_parallel_backtests,
    BacktestConfig,
    BacktestResult,
    generate_parameter_grid,
    create_sweep_configs,
    rank_results
)

__all__ = [
    "Backtester",
    "ForexConfig",
    "run_parallel_backtests",
    "BacktestConfig", 
    "BacktestResult",
    "generate_parameter_grid",
    "create_sweep_configs",
    "rank_results"
]
