"""
Monte Carlo Simulation Framework

Stress tests trading strategies by shuffling trade sequences to:
1. Calculate p-values (luck vs skill distinction)
2. Estimate probability of ruin under worst-case orderings
3. Provide confidence intervals for performance metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    strategy_name: str
    n_simulations: int
    original_sharpe: float
    original_return: float
    original_max_drawdown: float
    
    # Simulation results
    simulated_sharpes: List[float] = field(default_factory=list)
    simulated_returns: List[float] = field(default_factory=list)
    simulated_max_drawdowns: List[float] = field(default_factory=list)
    
    # Statistical metrics
    p_value: float = 0.0
    probability_of_ruin: float = 0.0
    sharpe_ci_lower: float = 0.0
    sharpe_ci_upper: float = 0.0
    
    @property
    def is_skill_based(self) -> bool:
        """Strategy shows skill if p < 0.05."""
        return self.p_value < 0.05
    
    @property
    def verdict(self) -> str:
        """Human-readable verdict."""
        if self.p_value < 0.01:
            return "STRONG_SKILL"
        elif self.p_value < 0.05:
            return "SKILL"
        elif self.p_value < 0.10:
            return "MARGINAL"
        else:
            return "LUCK"
    
    @property
    def sharpe_ci(self) -> Tuple[float, float]:
        """95% confidence interval for Sharpe."""
        return (self.sharpe_ci_lower, self.sharpe_ci_upper)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "strategy": self.strategy_name,
            "n_simulations": self.n_simulations,
            "original_sharpe": self.original_sharpe,
            "original_return": self.original_return,
            "original_max_drawdown": self.original_max_drawdown,
            "p_value": self.p_value,
            "probability_of_ruin": self.probability_of_ruin,
            "sharpe_ci": list(self.sharpe_ci),
            "verdict": self.verdict,
            "is_skill_based": self.is_skill_based,
            "simulated_sharpe_mean": float(np.mean(self.simulated_sharpes)) if self.simulated_sharpes else 0,
            "simulated_sharpe_std": float(np.std(self.simulated_sharpes)) if self.simulated_sharpes else 0,
        }


class MonteCarloSimulator:
    """
    Monte Carlo stress tester for trading strategies.
    
    Shuffles trade/return sequences to simulate alternative
    orderings and assess whether observed performance is due
    to genuine skill or favorable sequencing.
    """
    
    def __init__(
        self,
        n_simulations: int = 1000,
        ruin_threshold: float = -0.50,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of shuffle iterations
            ruin_threshold: Max drawdown threshold for ruin (-0.50 = 50%)
            confidence_level: Confidence level for intervals (0.95 = 95%)
            random_seed: Optional seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.ruin_threshold = ruin_threshold
        self.confidence_level = confidence_level
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def run_simulation(
        self,
        returns: np.ndarray,
        strategy_name: str = "Unknown"
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on a sequence of returns.
        
        Args:
            returns: Array of period returns (e.g., trade PnLs or bar returns)
            strategy_name: Name for identification
            
        Returns:
            MonteCarloResult with all simulation statistics
        """
        returns = np.asarray(returns)
        
        # Calculate original metrics
        original_sharpe = self._calculate_sharpe(returns)
        original_return = float(np.sum(returns))
        original_max_dd = self._calculate_max_drawdown(returns)
        
        # Run simulations
        simulated_sharpes = []
        simulated_returns = []
        simulated_max_dds = []
        ruin_count = 0
        
        for _ in range(self.n_simulations):
            # Shuffle returns (breaks temporal dependency)
            shuffled = np.random.permutation(returns)
            
            # Calculate metrics on shuffled sequence
            sim_sharpe = self._calculate_sharpe(shuffled)
            sim_return = float(np.sum(shuffled))
            sim_max_dd = self._calculate_max_drawdown(shuffled)
            
            simulated_sharpes.append(sim_sharpe)
            simulated_returns.append(sim_return)
            simulated_max_dds.append(sim_max_dd)
            
            # Check for ruin
            if sim_max_dd <= self.ruin_threshold:
                ruin_count += 1
        
        # Calculate p-value (one-tailed: how often random >= original)
        p_value = self._calculate_p_value(original_sharpe, simulated_sharpes)
        
        # Calculate confidence intervals
        ci_lower, ci_upper = self._calculate_confidence_interval(simulated_sharpes)
        
        # Probability of ruin
        prob_ruin = ruin_count / self.n_simulations
        
        return MonteCarloResult(
            strategy_name=strategy_name,
            n_simulations=self.n_simulations,
            original_sharpe=original_sharpe,
            original_return=original_return,
            original_max_drawdown=original_max_dd,
            simulated_sharpes=simulated_sharpes,
            simulated_returns=simulated_returns,
            simulated_max_drawdowns=simulated_max_dds,
            p_value=p_value,
            probability_of_ruin=prob_ruin,
            sharpe_ci_lower=ci_lower,
            sharpe_ci_upper=ci_upper
        )
    
    def run_from_trades(
        self,
        trades: List[Dict],
        strategy_name: str = "Unknown"
    ) -> MonteCarloResult:
        """
        Run Monte Carlo from trade-level data.
        
        Args:
            trades: List of trade dicts with 'pnl' key
            strategy_name: Name for identification
            
        Returns:
            MonteCarloResult
        """
        pnls = np.array([t.get("pnl", t.get("return", 0)) for t in trades])
        return self.run_simulation(pnls, strategy_name)
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Annualization factor (assuming minute data, 252 trading days)
        ann_factor = np.sqrt(252 * 24 * 60)
        
        return float(np.mean(returns) / np.std(returns) * ann_factor)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        if len(returns) == 0:
            return 0.0
        
        # Build equity curve
        equity = np.cumsum(returns)
        
        # Running maximum
        running_max = np.maximum.accumulate(equity)
        
        # Drawdown at each point
        drawdowns = equity - running_max
        
        # Return the worst (most negative) drawdown
        return float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    def _calculate_p_value(
        self,
        original: float,
        simulated: List[float]
    ) -> float:
        """
        Calculate p-value for skill vs luck test.
        
        H0: Strategy performance is random (no edge)
        H1: Strategy has genuine edge
        
        p-value = proportion of simulations >= original
        """
        if not simulated:
            return 1.0
        
        # Count how many simulated Sharpes >= original
        count_better_or_equal = sum(1 for s in simulated if s >= original)
        
        return count_better_or_equal / len(simulated)
    
    def _calculate_confidence_interval(
        self,
        values: List[float]
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval using percentile method.
        
        Returns:
            (lower_bound, upper_bound)
        """
        if not values:
            return (0.0, 0.0)
        
        alpha = 1 - self.confidence_level
        lower = np.percentile(values, alpha / 2 * 100)
        upper = np.percentile(values, (1 - alpha / 2) * 100)
        
        return (float(lower), float(upper))
    
    def analyze_sweep_results(
        self,
        sweep_file: Path,
        top_n: int = 5
    ) -> List[MonteCarloResult]:
        """
        Analyze top strategies from an existing sweep.
        
        Note: This method synthesizes returns from aggregate metrics.
        For precise analysis, use run_simulation with actual trade data.
        
        Args:
            sweep_file: Path to sweep_summary.json
            top_n: Number of top strategies to analyze
            
        Returns:
            List of MonteCarloResult for each strategy
        """
        with open(sweep_file) as f:
            results = json.load(f)
        
        # Sort by Sharpe and take top N
        sorted_results = sorted(
            results,
            key=lambda x: x.get("metrics", {}).get("sharpe", 0),
            reverse=True
        )[:top_n]
        
        mc_results = []
        
        for result in sorted_results:
            metrics = result.get("metrics", {})
            
            # Synthesize returns from aggregate metrics
            # This is an approximation - real trade data is preferred
            synthetic_returns = self._synthesize_returns(metrics)
            
            mc_result = self.run_simulation(
                returns=synthetic_returns,
                strategy_name=result.get("strategy", "Unknown")
            )
            
            mc_results.append(mc_result)
        
        return mc_results
    
    def _synthesize_returns(self, metrics: Dict) -> np.ndarray:
        """
        Synthesize return series from aggregate metrics.
        
        This is an approximation! For accurate Monte Carlo,
        use actual trade-level data.
        """
        total_trades = int(metrics.get("total_trades", 100))
        win_rate = metrics.get("win_rate", 0.5)
        total_return = metrics.get("total_return", 0)
        
        if total_trades <= 0:
            return np.array([0.0])
        
        # Estimate average win/loss size
        n_wins = int(total_trades * win_rate)
        n_losses = total_trades - n_wins
        
        if n_wins == 0:
            avg_win = 0
            avg_loss = total_return / max(n_losses, 1)
        elif n_losses == 0:
            avg_win = total_return / max(n_wins, 1)
            avg_loss = 0
        else:
            # Use profit factor to estimate relative win/loss sizes
            pf = metrics.get("profit_factor", 1.5)
            # avg_win * n_wins = pf * avg_loss * n_losses
            # avg_win * n_wins - avg_loss * n_losses = total_return
            
            # Solve for avg_win and avg_loss
            if pf > 0:
                avg_loss = total_return / (pf * n_wins - n_losses) if (pf * n_wins - n_losses) != 0 else 0.01
                avg_win = pf * avg_loss
            else:
                avg_win = total_return / n_wins if n_wins > 0 else 0
                avg_loss = 0
        
        # Generate synthetic returns
        returns = []
        for _ in range(n_wins):
            # Add some noise around avg_win
            returns.append(abs(avg_win) * (0.5 + np.random.random()))
        for _ in range(n_losses):
            # Add some noise around avg_loss
            returns.append(-abs(avg_loss) * (0.5 + np.random.random()))
        
        return np.array(returns)
    
    def save_results(
        self,
        results: List[MonteCarloResult],
        output_path: Path
    ):
        """Save Monte Carlo results to JSON."""
        data = [r.to_dict() for r in results]
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Monte Carlo results saved to {output_path}")


def print_mc_summary(results: List[MonteCarloResult]):
    """Print formatted Monte Carlo summary."""
    print("\n" + "=" * 65)
    print("MONTE CARLO STRESS TEST RESULTS")
    print("=" * 65)
    
    for r in results:
        verdict_icon = "✅" if r.is_skill_based else "⚠️"
        ruin_status = "🆘 HIGH RISK" if r.probability_of_ruin > 0.30 else ""
        
        print(f"\nStrategy: {r.strategy_name} (Sharpe: {r.original_sharpe:.2f})")
        print(f"  Simulations: {r.n_simulations:,}")
        print(f"  P-Value: {r.p_value:.4f} → {verdict_icon} {r.verdict}")
        print(f"  Probability of Ruin: {r.probability_of_ruin:.1%} {ruin_status}")
        print(f"  Sharpe 95% CI: [{r.sharpe_ci_lower:.2f}, {r.sharpe_ci_upper:.2f}]")
        print(f"  Simulated Sharpe Mean: {np.mean(r.simulated_sharpes):.2f} (±{np.std(r.simulated_sharpes):.2f})")


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation")
    parser.add_argument("--input", default="results/sweep_summary.json", help="Sweep results file")
    parser.add_argument("--simulations", type=int, default=1000, help="Number of simulations")
    parser.add_argument("--top", type=int, default=5, help="Top N strategies to analyze")
    parser.add_argument("--output", default="results/mc_results.json", help="Output file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    simulator = MonteCarloSimulator(
        n_simulations=args.simulations,
        random_seed=args.seed
    )
    
    print(f"Running Monte Carlo simulation ({args.simulations} iterations)...")
    print(f"Input: {args.input}")
    
    results = simulator.analyze_sweep_results(
        sweep_file=Path(args.input),
        top_n=args.top
    )
    
    # Print summary
    print_mc_summary(results)
    
    # Save results
    simulator.save_results(results, Path(args.output))
