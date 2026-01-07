"""
Walk-Forward Analysis Framework

Implements rolling window validation to detect overfitting and ensure
strategy robustness across different market regimes.
"""

from dataclasses import dataclass
from typing import Dict, List, Callable, Optional
from datetime import date, datetime
import polars as pl
import json
from pathlib import Path


@dataclass
class WalkForwardWindow:
    """Represents a single train/test window."""
    window_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_metrics: Optional[Dict] = None
    test_metrics: Optional[Dict] = None


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""
    strategy_name: str
    params: Dict
    windows: List[WalkForwardWindow]
    
    @property
    def test_sharpes(self) -> List[float]:
        return [w.test_metrics.get("sharpe", 0) for w in self.windows if w.test_metrics]
    
    @property
    def avg_test_sharpe(self) -> float:
        sharpes = self.test_sharpes
        return sum(sharpes) / len(sharpes) if sharpes else 0
    
    @property
    def consistency_ratio(self) -> float:
        """Ratio of windows with positive Sharpe."""
        sharpes = self.test_sharpes
        if not sharpes:
            return 0
        positive = sum(1 for s in sharpes if s > 0)
        return positive / len(sharpes)
    
    @property
    def is_robust(self) -> bool:
        """Strategy is robust if all windows have positive Sharpe."""
        return self.consistency_ratio == 1.0


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis engine.
    
    Uses rolling windows to validate strategy performance across
    different time periods, detecting overfitting.
    """
    
    # Default windows based on 22 years of data (2003-2024)
    DEFAULT_WINDOWS = [
        {"train": (2003, 2012), "test": (2012, 2014)},
        {"train": (2005, 2014), "test": (2014, 2016)},
        {"train": (2007, 2016), "test": (2016, 2018)},
        {"train": (2009, 2018), "test": (2018, 2020)},
        {"train": (2011, 2020), "test": (2020, 2022)},
        {"train": (2013, 2022), "test": (2022, 2024)},
    ]
    
    def __init__(self, data_path: Path, windows: Optional[List[Dict]] = None):
        """
        Initialize analyzer.
        
        Args:
            data_path: Path to parquet data directory
            windows: Custom train/test windows (optional)
        """
        self.data_path = Path(data_path)
        self.windows = windows or self.DEFAULT_WINDOWS
    
    def load_data_for_period(self, pair: str, start_year: int, end_year: int) -> pl.DataFrame:
        """Load data for a specific year range."""
        frames = []
        for year in range(start_year, end_year + 1):
            path = self.data_path / pair / f"{year}.parquet"
            if path.exists():
                frames.append(pl.read_parquet(path))
        
        if not frames:
            raise FileNotFoundError(f"No data found for {pair} in {start_year}-{end_year}")
        
        return pl.concat(frames)
    
    def run_analysis(
        self,
        strategy_class: type,
        params: Dict,
        pair: str = "XAUUSD"
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis on a strategy.
        
        Args:
            strategy_class: Strategy class to test
            params: Strategy parameters
            pair: Currency pair
            
        Returns:
            WalkForwardResult with all window metrics
        """
        windows = []
        
        for i, window_config in enumerate(self.windows):
            train_years = window_config["train"]
            test_years = window_config["test"]
            
            window = WalkForwardWindow(
                window_id=i + 1,
                train_start=date(train_years[0], 1, 1),
                train_end=date(train_years[1], 12, 31),
                test_start=date(test_years[0], 1, 1),
                test_end=date(test_years[1], 12, 31),
            )
            
            try:
                # Load and run on training period
                train_data = self.load_data_for_period(pair, *train_years)
                strategy = strategy_class(**params)
                window.train_metrics = strategy.backtest(train_data)
                
                # Load and run on test period
                test_data = self.load_data_for_period(pair, *test_years)
                window.test_metrics = strategy.backtest(test_data)
                
            except FileNotFoundError as e:
                print(f"Window {i+1} skipped: {e}")
                window.train_metrics = {"error": str(e)}
                window.test_metrics = {"error": str(e)}
            except Exception as e:
                print(f"Window {i+1} error: {e}")
                window.train_metrics = {"error": str(e)}
                window.test_metrics = {"error": str(e)}
            
            windows.append(window)
        
        return WalkForwardResult(
            strategy_name=strategy_class.__name__,
            params=params,
            windows=windows
        )
    
    def run_full_sweep(
        self,
        strategy_configs: List[Dict],
        pair: str = "XAUUSD"
    ) -> List[WalkForwardResult]:
        """
        Run walk-forward analysis on multiple strategy configurations.
        
        Args:
            strategy_configs: List of {"strategy_class": class, "params": dict}
            pair: Currency pair
            
        Returns:
            List of WalkForwardResult for each configuration
        """
        results = []
        
        for config in strategy_configs:
            result = self.run_analysis(
                config["strategy_class"],
                config["params"],
                pair
            )
            results.append(result)
            
            # Print progress
            status = "✅ ROBUST" if result.is_robust else f"⚠️ {result.consistency_ratio:.0%}"
            print(f"{result.strategy_name}: Avg Test Sharpe={result.avg_test_sharpe:.2f} {status}")
        
        return results
    
    def save_results(self, results: List[WalkForwardResult], output_path: Path):
        """Save results to JSON."""
        data = []
        for r in results:
            data.append({
                "strategy": r.strategy_name,
                "params": r.params,
                "avg_test_sharpe": r.avg_test_sharpe,
                "consistency_ratio": r.consistency_ratio,
                "is_robust": r.is_robust,
                "windows": [
                    {
                        "window_id": w.window_id,
                        "train_period": f"{w.train_start.year}-{w.train_end.year}",
                        "test_period": f"{w.test_start.year}-{w.test_end.year}",
                        "train_sharpe": w.train_metrics.get("sharpe", 0) if w.train_metrics else 0,
                        "test_sharpe": w.test_metrics.get("sharpe", 0) if w.test_metrics else 0,
                    }
                    for w in r.windows
                ]
            })
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {output_path}")


def generate_adaptive_windows(
    start_year: int = 2010,
    end_year: int = 2024,
    train_years: int = 5,
    test_years: int = 2,
    step_years: int = 2
) -> List[Dict]:
    """
    Generate adaptive walk-forward windows based on available data.
    
    Args:
        start_year: First year of available data
        end_year: Last year of available data
        train_years: Training period length
        test_years: Testing period length
        step_years: Years to step forward each window
        
    Returns:
        List of window configurations
    """
    windows = []
    current_start = start_year
    
    while current_start + train_years + test_years <= end_year + 1:
        train_end = current_start + train_years
        test_end = train_end + test_years
        
        windows.append({
            "train": (current_start, train_end - 1),
            "test": (train_end, test_end - 1)
        })
        
        current_start += step_years
    
    return windows


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Walk-Forward Analysis")
    parser.add_argument("--data-path", default="data/parquet", help="Data directory")
    parser.add_argument("--pair", default="XAUUSD", help="Currency pair")
    parser.add_argument("--output", default="results/wfa_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Generate windows for available data (2010-2024)
    windows = generate_adaptive_windows(2010, 2024, train_years=3, test_years=1, step_years=2)
    
    print(f"Generated {len(windows)} walk-forward windows:")
    for w in windows:
        print(f"  Train: {w['train'][0]}-{w['train'][1]} → Test: {w['test'][0]}-{w['test'][1]}")
    
    analyzer = WalkForwardAnalyzer(args.data_path, windows)
    
    # Import and test a sample strategy
    from strategies.pattern_based.candlestick import Strategy as CandlestickStrategy
    
    result = analyzer.run_analysis(
        CandlestickStrategy,
        {"patterns": ["doji", "engulfing"], "confirm_period": 3},
        args.pair
    )
    
    print(f"\nResults for {result.strategy_name}:")
    print(f"  Average Test Sharpe: {result.avg_test_sharpe:.2f}")
    print(f"  Consistency Ratio: {result.consistency_ratio:.0%}")
    print(f"  Is Robust: {result.is_robust}")
