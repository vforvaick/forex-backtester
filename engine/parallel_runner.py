"""
Parallel Backtesting Runner

Runs multiple backtests in parallel using joblib.
Optimized for distributed execution across MacBook, thinktank, and VPS.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import importlib

from joblib import Parallel, delayed
import polars as pl

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from journal import save_backtest_run


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""
    strategy_name: str
    strategy_module: str  # e.g., "strategies.trend_following.moving_average"
    params: Dict[str, Any]
    data_path: Path
    start_date: str
    end_date: str


@dataclass
class BacktestResult:
    """Result from a single backtest run."""
    config: BacktestConfig
    metrics: Dict[str, float]
    run_id: int
    duration_seconds: float
    success: bool
    error: Optional[str] = None


def run_single_backtest(config: BacktestConfig) -> BacktestResult:
    """
    Run a single backtest with given configuration.
    
    This function is designed to be called in parallel.
    """
    import time
    start_time = time.time()
    
    try:
        from engine.backtester import Backtester, ForexConfig
        
        # Initialize backtester config
        forex_config = ForexConfig(symbol=config.strategy_name.split("_")[0]) # Rough guess
        backtester = Backtester(forex_config)
        
        # Load strategy module
        module = importlib.import_module(config.strategy_module)
        # Find Strategy class (might be in strategy_module or its children)
        # Assuming most strategies follow the moving_average_crossover pattern
        if hasattr(module, "Strategy"):
            strategy_class = getattr(module, "Strategy")
        else:
            # Try to find class that inherits from something or just look for 'Strategy'
            raise AttributeError(f"Module {config.strategy_module} has no 'Strategy' class")
        
        # Load data
        data = backtester.load_data(config.data_path, config.start_date, config.end_date)
        
        if data.is_empty():
            raise ValueError(f"No data found for {config.start_date} to {config.end_date}")

        # Strategy instance returns full metrics via backtest()
        def strategy_func(df, p):
            s = strategy_class(**p)
            metrics = s.backtest(df)
            return metrics

        # Run backtest - now returns metrics directly
        metrics = backtester.run(strategy_func, data, config.params)
        
        # Save to journal
        run_id = save_backtest_run(
            strategy_name=config.strategy_name,
            tuning_params=config.params,
            metrics=metrics,
            data_range=(config.start_date, config.end_date),
            verdict="promising" if metrics.get("sharpe", 0) > 1.5 else "needs_refinement"
        )
        
        duration = time.time() - start_time
        
        return BacktestResult(
            config=config,
            metrics=metrics,
            run_id=run_id,
            duration_seconds=duration,
            success=True
        )
        
    except Exception as e:
        import traceback
        print(f"Error in {config.strategy_name}: {e}")
        # traceback.print_exc()
        duration = time.time() - start_time
        return BacktestResult(
            config=config,
            metrics={},
            run_id=-1,
            duration_seconds=duration,
            success=False,
            error=str(e)
        )


def run_parallel_backtests(
    configs: List[BacktestConfig],
    n_jobs: int = -1,
    verbose: int = 10
) -> List[BacktestResult]:
    """
    Run multiple backtests in parallel.
    
    Args:
        configs: List of backtest configurations
        n_jobs: Number of parallel jobs (-1 = all cores)
        verbose: Verbosity level for progress
    
    Returns:
        List of BacktestResult objects
    """
    print(f"Running {len(configs)} backtests with {n_jobs} parallel jobs...")
    
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_single_backtest)(config) for config in configs
    )
    
    # Summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    total_time = sum(r.duration_seconds for r in results)
    
    print(f"\nCompleted: {successful} successful, {failed} failed")
    print(f"Total compute time: {total_time:.1f}s")
    
    return results


def generate_parameter_grid(
    base_strategy: str,
    param_ranges: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations from ranges.
    
    Example:
        param_ranges = {
            "period": [10, 20, 50],
            "threshold": [0.5, 1.0]
        }
        Returns 6 combinations
    """
    from itertools import product
    
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())
    
    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def create_sweep_configs(
    strategy_name: str,
    strategy_module: str,
    param_ranges: Dict[str, List[Any]],
    data_path: Path,
    start_date: str,
    end_date: str
) -> List[BacktestConfig]:
    """Create BacktestConfig for each parameter combination."""
    
    param_combos = generate_parameter_grid(strategy_name, param_ranges)
    
    configs = []
    for i, params in enumerate(param_combos):
        configs.append(BacktestConfig(
            strategy_name=f"{strategy_name}_v{i:03d}",
            strategy_module=strategy_module,
            params=params,
            data_path=data_path,
            start_date=start_date,
            end_date=end_date
        ))
    
    return configs


def rank_results(results: List[BacktestResult], metric: str = "sharpe") -> List[BacktestResult]:
    """Rank results by specified metric (descending)."""
    valid = [r for r in results if r.success and r.metrics.get(metric) is not None]
    return sorted(valid, key=lambda r: r.metrics.get(metric, 0), reverse=True)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run parallel backtests")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs")
    parser.add_argument("--output", default="results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Load config and run (placeholder)
    print(f"Would run with config: {args.config}, jobs: {args.n_jobs}")
