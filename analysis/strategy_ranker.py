"""
Strategy Ranker Module

Provides composite scoring and ranking for trading strategies.
Uses normalized metrics with configurable weights.
"""

import polars as pl
from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class RankingConfig:
    """Configuration for strategy ranking."""
    weights: dict[str, float]
    min_trades: int = 100
    max_drawdown_cap: float = 0.50
    min_profit_factor: float = 1.0
    
    @classmethod
    def from_yaml(cls, path: str = "config/optimization_policy.yaml") -> "RankingConfig":
        """Load ranking config from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        
        # Use ranking_weights if present, else fall back to weights
        weights = config.get("ranking_weights", config.get("weights", {}))
        constraints = config.get("constraints", {})
        
        return cls(
            weights=weights,
            min_trades=constraints.get("min_trades", 100),
            max_drawdown_cap=constraints.get("max_drawdown", 0.50),
            min_profit_factor=constraints.get("min_profit_factor", 1.0)
        )


@dataclass 
class RankedStrategy:
    """A strategy with its composite score."""
    strategy_name: str
    params: dict
    composite_score: float
    metrics: dict
    passed_filters: bool
    filter_failures: list[str]


class StrategyRanker:
    """
    Ranks strategies using a composite score formula.
    
    Score = Σ (weight_i × normalized_metric_i)
    
    Default weights (approved):
        sharpe: 0.25
        sortino: 0.15
        calmar: 0.20
        profit_factor: 0.15
        max_drawdown: 0.10 (inverted)
        win_rate: 0.05
        recovery_factor: 0.10
    """
    
    DEFAULT_WEIGHTS = {
        "sharpe": 0.25,
        "sortino": 0.15,
        "calmar": 0.20,
        "profit_factor": 0.15,
        "max_drawdown": 0.10,
        "win_rate": 0.05,
        "recovery_factor": 0.10
    }
    
    def __init__(self, config: Optional[RankingConfig] = None):
        """Initialize ranker with optional config."""
        self.config = config or RankingConfig(weights=self.DEFAULT_WEIGHTS)
        self.weights = {**self.DEFAULT_WEIGHTS, **self.config.weights}
    
    def normalize(self, value: float, min_val: float, max_val: float, invert: bool = False) -> float:
        """Normalize value to 0-1 range."""
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
        
        if invert:
            normalized = 1.0 - normalized
        
        return normalized
    
    def check_filters(self, metrics: dict) -> tuple[bool, list[str]]:
        """Check if strategy passes all filters."""
        failures = []
        
        if metrics.get("total_trades", 0) < self.config.min_trades:
            failures.append(f"trades={metrics.get('total_trades', 0)} < {self.config.min_trades}")
        
        if abs(metrics.get("max_drawdown", 1.0)) > self.config.max_drawdown_cap:
            failures.append(f"drawdown={abs(metrics.get('max_drawdown', 1.0)):.2%} > {self.config.max_drawdown_cap:.0%}")
        
        if metrics.get("profit_factor", 0) < self.config.min_profit_factor:
            failures.append(f"PF={metrics.get('profit_factor', 0):.2f} < {self.config.min_profit_factor}")
        
        return len(failures) == 0, failures
    
    def compute_score(self, metrics: dict, bounds: Optional[dict] = None) -> float:
        """
        Compute composite score for a single strategy.
        
        Args:
            metrics: Dict with sharpe, sortino, calmar, profit_factor, max_drawdown, win_rate, recovery_factor
            bounds: Optional dict with min/max for each metric (for cross-strategy normalization)
        
        Returns:
            Composite score between 0 and 1
        """
        # Default bounds if not provided (reasonable trading ranges)
        default_bounds = {
            "sharpe": (-3.0, 3.0),
            "sortino": (-3.0, 5.0),
            "calmar": (-2.0, 5.0),
            "profit_factor": (0.0, 3.0),
            "max_drawdown": (0.0, 1.0),  # Will be inverted
            "win_rate": (0.0, 1.0),
            "recovery_factor": (0.0, 10.0)
        }
        bounds = bounds or default_bounds
        
        score = 0.0
        
        for metric, weight in self.weights.items():
            value = metrics.get(metric, 0)
            
            # Handle max_drawdown specially (lower is better)
            if metric == "max_drawdown":
                value = abs(value)  # Ensure positive
                min_val, max_val = bounds.get(metric, (0.0, 1.0))
                normalized = self.normalize(value, min_val, max_val, invert=True)
            else:
                min_val, max_val = bounds.get(metric, (0.0, 1.0))
                normalized = self.normalize(value, min_val, max_val)
            
            score += weight * normalized
        
        return score
    
    def rank_strategies(self, results: list[dict]) -> list[RankedStrategy]:
        """
        Rank a list of strategy results.
        
        Args:
            results: List of dicts with 'strategy_name', 'params', and metric keys
        
        Returns:
            Sorted list of RankedStrategy (highest score first)
        """
        # First, compute bounds across all strategies for fair normalization
        bounds = {}
        metric_keys = ["sharpe", "sortino", "calmar", "profit_factor", "max_drawdown", "win_rate", "recovery_factor"]
        
        for key in metric_keys:
            values = [r.get(key, 0) for r in results if key in r]
            if values:
                bounds[key] = (min(values), max(values))
        
        # Score and rank
        ranked = []
        for result in results:
            passed, failures = self.check_filters(result)
            score = self.compute_score(result, bounds) if passed else 0.0
            
            ranked.append(RankedStrategy(
                strategy_name=result.get("strategy_name", "unknown"),
                params=result.get("params", {}),
                composite_score=score,
                metrics={k: result.get(k) for k in metric_keys if k in result},
                passed_filters=passed,
                filter_failures=failures
            ))
        
        # Sort by score (descending), with filter-failed strategies at the end
        ranked.sort(key=lambda x: (x.passed_filters, x.composite_score), reverse=True)
        
        return ranked
    
    def rank_from_journal(self, db_path: str = "backtest_journal.db") -> list[RankedStrategy]:
        """
        Load results from journal database and rank them.
        
        Args:
            db_path: Path to SQLite journal database
        
        Returns:
            Sorted list of RankedStrategy
        """
        import sqlite3
        import json
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT strategy_name, params, metrics 
            FROM backtest_runs 
            WHERE status = 'success'
        """)
        
        results = []
        for row in cursor.fetchall():
            strategy_name, params_json, metrics_json = row
            params = json.loads(params_json) if params_json else {}
            metrics = json.loads(metrics_json) if metrics_json else {}
            
            results.append({
                "strategy_name": strategy_name,
                "params": params,
                **metrics
            })
        
        conn.close()
        
        return self.rank_strategies(results)
    
    def print_ranking(self, ranked: list[RankedStrategy], top_n: int = 10) -> None:
        """Print formatted ranking table."""
        print(f"\n{'='*80}")
        print(f"{'STRATEGY RANKING':^80}")
        print(f"{'='*80}")
        print(f"{'Rank':<6}{'Strategy':<30}{'Score':<10}{'Sharpe':<10}{'DD%':<10}{'Status':<15}")
        print(f"{'-'*80}")
        
        for i, r in enumerate(ranked[:top_n], 1):
            status = "✓ PASS" if r.passed_filters else f"✗ {r.filter_failures[0][:12]}"
            dd = abs(r.metrics.get("max_drawdown", 0)) * 100
            
            print(f"{i:<6}{r.strategy_name[:28]:<30}{r.composite_score:<10.4f}"
                  f"{r.metrics.get('sharpe', 0):<10.2f}{dd:<10.1f}{status:<15}")
        
        print(f"{'='*80}\n")


def rank_strategies(results: list[dict], config_path: Optional[str] = None) -> list[RankedStrategy]:
    """
    Convenience function to rank strategies.
    
    Args:
        results: List of strategy result dicts
        config_path: Optional path to config YAML
    
    Returns:
        Sorted list of RankedStrategy
    """
    config = RankingConfig.from_yaml(config_path) if config_path else None
    ranker = StrategyRanker(config)
    return ranker.rank_strategies(results)


if __name__ == "__main__":
    # Demo usage
    sample_results = [
        {
            "strategy_name": "ma_crossover_fast",
            "params": {"fast": 5, "slow": 20},
            "sharpe": 1.2,
            "sortino": 1.8,
            "calmar": 0.9,
            "profit_factor": 1.5,
            "max_drawdown": 0.15,
            "win_rate": 0.52,
            "recovery_factor": 3.2,
            "total_trades": 250
        },
        {
            "strategy_name": "bollinger_bands",
            "params": {"period": 20, "std": 2.0},
            "sharpe": 0.8,
            "sortino": 1.1,
            "calmar": 0.6,
            "profit_factor": 1.3,
            "max_drawdown": 0.22,
            "win_rate": 0.48,
            "recovery_factor": 2.1,
            "total_trades": 180
        },
        {
            "strategy_name": "rsi_oversold",
            "params": {"period": 14},
            "sharpe": 0.3,
            "sortino": 0.4,
            "calmar": 0.2,
            "profit_factor": 0.9,  # Will fail filter
            "max_drawdown": 0.35,
            "win_rate": 0.45,
            "recovery_factor": 0.8,
            "total_trades": 120
        }
    ]
    
    ranker = StrategyRanker()
    ranked = ranker.rank_strategies(sample_results)
    ranker.print_ranking(ranked)
