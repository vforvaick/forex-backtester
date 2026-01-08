"""
Parameter Recommender

Handles:
- Multi-model consensus checking
- Parameter bounds validation
- Trial config generation with _meta
- Monte Carlo stress test integration
- A/B comparison with confidence intervals
"""

import json
import yaml
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.monte_carlo import MonteCarloSimulator


@dataclass
class TrialMeta:
    """Metadata for a trial configuration."""
    trial_id: str
    strategy: str
    category: str
    source_run_id: int
    llm_model: str
    created_at: str
    changes: List[Dict[str, Any]] = field(default_factory=list)
    consensus_models: List[str] = field(default_factory=list)


@dataclass
class TrialConfig:
    """A complete trial configuration."""
    meta: TrialMeta
    params: Dict[str, Any]
    original_params: Dict[str, Any]


class ParameterRecommender:
    """
    Manages the parameter recommendation loop.
    
    Features:
    - Multi-model consensus (2/3 agreement)
    - Bounds validation
    - Trial generation with versioning
    - Monte Carlo stress testing
    - Policy-based comparison
    """
    
    def __init__(
        self,
        tuning_grid_path: Path = Path("config/tuning_grid.yaml"),
        policy_path: Path = Path("config/optimization_policy.yaml"),
        trials_dir: Path = Path("trials")
    ):
        self.tuning_grid_path = tuning_grid_path
        self.policy_path = policy_path
        self.trials_dir = trials_dir
        
        # Load configs
        with open(tuning_grid_path) as f:
            self.tuning_grid = yaml.safe_load(f)
        
        with open(policy_path) as f:
            self.policy = yaml.safe_load(f)
    
    def get_bounds(self, category: str, strategy: str) -> Optional[Dict[str, List]]:
        """Get parameter bounds for a strategy."""
        bounds = self.tuning_grid.get(category, {}).get("_bounds", {})
        return bounds.get(strategy)
    
    def validate_params(self, category: str, strategy: str, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameters against bounds.
        
        Returns:
            (is_valid, list of errors)
        """
        bounds = self.get_bounds(category, strategy)
        if not bounds:
            return True, []  # No bounds defined, accept
        
        errors = []
        for param, value in params.items():
            if param in bounds:
                min_val, max_val = bounds[param]
                if not (min_val <= value <= max_val):
                    errors.append(f"{param}={value} out of range [{min_val}, {max_val}]")
        
        return len(errors) == 0, errors
    
    def check_consensus(self, evaluations: List[Dict]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if 2/3 models agree on parameter recommendations.
        
        Returns:
            (has_consensus, agreed_recommendations)
        """
        threshold = self.policy.get("consensus", {}).get("threshold", 0.66)
        
        # Extract recommendations from each model
        all_recs = {}
        for eval_result in evaluations:
            model = eval_result.get("model", "unknown")
            recs = eval_result.get("recommendations", [])
            for rec in recs:
                param = rec.get("param")
                suggested = rec.get("suggested")
                if param:
                    if param not in all_recs:
                        all_recs[param] = []
                    all_recs[param].append({
                        "model": model,
                        "suggested": suggested,
                        "reasoning": rec.get("reasoning", "")
                    })
        
        # Find consensus
        agreed = {}
        models_agreed = set()
        
        for param, suggestions in all_recs.items():
            # Group by suggested value
            value_counts = {}
            for s in suggestions:
                val = s["suggested"]
                if val not in value_counts:
                    value_counts[val] = []
                value_counts[val].append(s)
            
            # Check if any value has enough agreement
            for val, supporters in value_counts.items():
                agreement_ratio = len(supporters) / len(evaluations)
                if agreement_ratio >= threshold:
                    agreed[param] = {
                        "suggested": val,
                        "reasoning": supporters[0]["reasoning"],
                        "agreement": agreement_ratio,
                        "models": [s["model"] for s in supporters]
                    }
                    models_agreed.update([s["model"] for s in supporters])
        
        has_consensus = len(agreed) > 0
        return has_consensus, agreed
    
    def generate_trial(
        self,
        category: str,
        strategy: str,
        recommendations: Dict[str, Any],
        source_run_id: int,
        llm_model: str
    ) -> Optional[TrialConfig]:
        """
        Generate a trial config from recommendations.
        
        Creates trials/{category}/{strategy}/trial_XXX_TIMESTAMP.yaml
        """
        # Get current params (first variant as baseline)
        variants = self.tuning_grid.get(category, {}).get(strategy, [])
        if not variants:
            return None
        
        original_params = variants[0].copy()
        
        # Apply recommendations
        new_params = original_params.copy()
        changes = []
        consensus_models = []
        
        for param, rec in recommendations.items():
            if param in new_params:
                old_val = new_params[param]
                new_val = rec["suggested"]
                new_params[param] = new_val
                changes.append({
                    "param": param,
                    "old": old_val,
                    "new": new_val,
                    "reasoning": rec["reasoning"]
                })
                if "models" in rec:
                    consensus_models.extend(rec["models"])
        
        # Validate
        is_valid, errors = self.validate_params(category, strategy, new_params)
        if not is_valid:
            print(f"❌ Invalid params: {errors}")
            return None
        
        # Generate trial ID
        trial_dir = self.trials_dir / category / strategy
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        existing = list(trial_dir.glob("trial_*.yaml"))
        trial_num = len(existing) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_id = f"trial_{trial_num:03d}_{timestamp}"
        
        # Create meta
        meta = TrialMeta(
            trial_id=trial_id,
            strategy=strategy,
            category=category,
            source_run_id=source_run_id,
            llm_model=llm_model,
            created_at=datetime.now().isoformat(),
            changes=changes,
            consensus_models=list(set(consensus_models))
        )
        
        trial = TrialConfig(
            meta=meta,
            params=new_params,
            original_params=original_params
        )
        
        # Save to file
        trial_path = trial_dir / f"{trial_id}.yaml"
        with open(trial_path, "w") as f:
            yaml.dump({
                "_meta": asdict(meta),
                "params": new_params,
                "original_params": original_params
            }, f, default_flow_style=False)
        
        print(f"✅ Generated trial: {trial_path}")
        return trial
    
    def run_mc_check(self, metrics: Dict[str, float], n_simulations: int = 1000) -> Dict[str, float]:
        """
        Run Monte Carlo stress test on trial results.
        
        Returns:
            {p_value, ruin_prob, sharpe_ci_low, sharpe_ci_high}
        """
        simulator = MonteCarloSimulator(n_simulations=n_simulations)
        
        # Extract trade returns if available, else estimate
        trades = metrics.get("total_trades", 100)
        win_rate = metrics.get("win_rate", 0.5)
        
        # Simulate trade sequence
        import numpy as np
        np.random.seed(42)
        
        # Estimate avg win/loss from profit factor
        pf = metrics.get("profit_factor", 1.5)
        avg_win = 0.02  # 2% avg win
        avg_loss = -avg_win / pf if pf > 0 else -0.02
        
        # Generate synthetic trades
        wins = int(trades * win_rate)
        losses = trades - wins
        trade_returns = list(np.random.normal(avg_win, 0.01, wins)) + \
                       list(np.random.normal(avg_loss, 0.01, losses))
        
        # Run MC
        result = simulator.run_simulation(trade_returns)
        
        return {
            "p_value": result.p_value,
            "ruin_prob": result.probability_of_ruin,
            "sharpe_ci_low": result.sharpe_ci[0],
            "sharpe_ci_high": result.sharpe_ci[1],
            "confidence": 1 - result.p_value
        }
    
    def compute_score(self, metrics: Dict[str, float]) -> Tuple[float, bool, List[str]]:
        """
        Compute weighted score and check constraints.
        
        Returns:
            (score, passes_constraints, violations)
        """
        weights = self.policy.get("weights", {})
        constraints = self.policy.get("constraints", {})
        
        # Compute weighted score
        score = 0.0
        for metric, weight in weights.items():
            val = metrics.get(metric, 0)
            score += weight * val
        
        # Check constraints
        violations = []
        
        if metrics.get("total_trades", 0) < constraints.get("min_trades", 0):
            violations.append(f"trades {metrics.get('total_trades')} < min {constraints.get('min_trades')}")
        
        if abs(metrics.get("max_drawdown", 0)) > constraints.get("max_drawdown", 1.0):
            violations.append(f"drawdown {metrics.get('max_drawdown'):.1%} > max {constraints.get('max_drawdown'):.1%}")
        
        if metrics.get("win_rate", 0) < constraints.get("min_win_rate", 0):
            violations.append(f"win_rate {metrics.get('win_rate'):.1%} < min {constraints.get('min_win_rate'):.1%}")
        
        passes = len(violations) == 0
        return score, passes, violations
    
    def compare(
        self,
        original_metrics: Dict[str, float],
        trial_metrics: Dict[str, float],
        original_mc: Optional[Dict] = None,
        trial_mc: Optional[Dict] = None
    ) -> str:
        """
        Generate A/B comparison table.
        """
        # Compute scores
        orig_score, orig_passes, orig_violations = self.compute_score(original_metrics)
        trial_score, trial_passes, trial_violations = self.compute_score(trial_metrics)
        
        lines = []
        lines.append("┌─────────────────────┬──────────┬───────────────┬────────────┐")
        lines.append("│ Metric              │ Original │ Trial         │ Confidence │")
        lines.append("├─────────────────────┼──────────┼───────────────┼────────────┤")
        
        def fmt_row(name: str, orig: float, trial: float, fmt: str = ".2f", conf: str = "-"):
            better = trial > orig if name not in ["max_drawdown", "ruin_prob", "p_value"] else trial < orig
            marker = "✅" if better else "⚠️" if trial != orig else ""
            return f"│ {name:<19} │ {orig:{fmt}:<8} │ {trial:{fmt}:<7} {marker:<4} │ {conf:<10} │"
        
        # Net Sharpe (with costs)
        orig_net = original_metrics.get("sharpe", 0) * (1 - original_metrics.get("cost_impact", 0.05))
        trial_net = trial_metrics.get("sharpe", 0) * (1 - trial_metrics.get("cost_impact", 0.05))
        conf_str = f"{trial_mc.get('confidence', 0):.0%}" if trial_mc else "-"
        lines.append(fmt_row("Net Sharpe", orig_net, trial_net, conf=conf_str))
        
        lines.append(fmt_row("Gross Sharpe", original_metrics.get("sharpe", 0), trial_metrics.get("sharpe", 0)))
        lines.append(fmt_row("Max Drawdown", original_metrics.get("max_drawdown", 0), trial_metrics.get("max_drawdown", 0), fmt=".1%"))
        lines.append(fmt_row("Win Rate", original_metrics.get("win_rate", 0), trial_metrics.get("win_rate", 0), fmt=".1%"))
        lines.append(fmt_row("Profit Factor", original_metrics.get("profit_factor", 0), trial_metrics.get("profit_factor", 0)))
        lines.append(fmt_row("Total Trades", original_metrics.get("total_trades", 0), trial_metrics.get("total_trades", 0), fmt=".0f"))
        
        # MC metrics
        if original_mc and trial_mc:
            lines.append("├─────────────────────┼──────────┼───────────────┼────────────┤")
            lines.append(fmt_row("MC P-Value", original_mc.get("p_value", 0), trial_mc.get("p_value", 0)))
            lines.append(fmt_row("Ruin Prob", original_mc.get("ruin_prob", 0), trial_mc.get("ruin_prob", 0), fmt=".1%"))
        
        lines.append("├─────────────────────┼──────────┼───────────────┼────────────┤")
        lines.append(f"│ {'Policy Score':<19} │ {orig_score:<8.2f} │ {trial_score:<7.2f} {'✅' if trial_score > orig_score else '':<4} │ {'-':<10} │")
        lines.append("└─────────────────────┴──────────┴───────────────┴────────────┘")
        
        # Violations
        if trial_violations:
            lines.append("\n⚠️ Trial Violations:")
            for v in trial_violations:
                lines.append(f"  - {v}")
        
        return "\n".join(lines)
    
    def explain_trial(self, trial_path: Path) -> str:
        """
        Generate human-readable explanation of a trial.
        """
        with open(trial_path) as f:
            trial = yaml.safe_load(f)
        
        meta = trial.get("_meta", {})
        lines = []
        lines.append(f"Trial: {meta.get('trial_id')}")
        lines.append(f"Strategy: {meta.get('category')}/{meta.get('strategy')}")
        lines.append(f"Created: {meta.get('created_at')}")
        lines.append(f"Source Run: {meta.get('source_run_id')}")
        lines.append(f"LLM Model: {meta.get('llm_model')}")
        lines.append(f"Consensus Models: {', '.join(meta.get('consensus_models', []))}")
        lines.append("")
        lines.append("Changes:")
        
        for i, change in enumerate(meta.get("changes", []), 1):
            lines.append(f"  {i}. {change['param']}: {change['old']} → {change['new']}")
            lines.append(f"     Reason: \"{change['reasoning']}\"")
        
        return "\n".join(lines)
    
    def git_promote(self, trial_path: Path, dry_run: bool = False) -> Tuple[bool, str]:
        """
        Promote trial to tuning_grid.yaml with git commit.
        
        Returns:
            (success, message)
        """
        with open(trial_path) as f:
            trial = yaml.safe_load(f)
        
        meta = trial.get("_meta", {})
        category = meta.get("category")
        strategy = meta.get("strategy")
        new_params = trial.get("params", {})
        
        if dry_run:
            lines = ["Would update tuning_grid.yaml:"]
            for change in meta.get("changes", []):
                lines.append(f"  {strategy}.{change['param']}: {change['old']} → {change['new']}")
            lines.append("\nNo changes made (dry-run).")
            return True, "\n".join(lines)
        
        # Update tuning_grid.yaml
        with open(self.tuning_grid_path) as f:
            grid = yaml.safe_load(f)
        
        # Replace first variant
        if category in grid and strategy in grid[category]:
            variants = grid[category][strategy]
            if isinstance(variants, list) and len(variants) > 0:
                grid[category][strategy][0] = new_params
        
        with open(self.tuning_grid_path, "w") as f:
            yaml.dump(grid, f, default_flow_style=False)
        
        # Git commit
        trial_id = meta.get("trial_id")
        commit_msg = f"optimize: promote {strategy} {trial_id}"
        
        try:
            subprocess.run(["git", "add", str(self.tuning_grid_path)], check=True, cwd=self.tuning_grid_path.parent)
            subprocess.run(["git", "commit", "-m", commit_msg], check=True, cwd=self.tuning_grid_path.parent)
            return True, f"✅ Promoted {trial_id} and committed: {commit_msg}"
        except subprocess.CalledProcessError as e:
            return False, f"❌ Git error: {e}"
    
    def git_undo(self, trial_id: str, dry_run: bool = False) -> Tuple[bool, str]:
        """
        Undo a promotion by reverting the git commit.
        """
        # Find the commit
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--grep", f"promote.*{trial_id}"],
                capture_output=True, text=True, cwd=self.tuning_grid_path.parent
            )
            commits = result.stdout.strip().split("\n")
            if not commits or not commits[0]:
                return False, f"No promotion commit found for {trial_id}"
            
            commit_hash = commits[0].split()[0]
            
            if dry_run:
                # Show what would be reverted
                diff_result = subprocess.run(
                    ["git", "show", "--stat", commit_hash],
                    capture_output=True, text=True, cwd=self.tuning_grid_path.parent
                )
                return True, f"Would revert commit {commit_hash}:\n{diff_result.stdout[:500]}"
            
            # Actually revert
            subprocess.run(["git", "revert", "--no-commit", commit_hash], check=True, cwd=self.tuning_grid_path.parent)
            subprocess.run(["git", "commit", "-m", f"Revert: undo {trial_id}"], check=True, cwd=self.tuning_grid_path.parent)
            return True, f"✅ Reverted {trial_id}"
            
        except subprocess.CalledProcessError as e:
            return False, f"❌ Git error: {e}"


# CLI test
if __name__ == "__main__":
    recommender = ParameterRecommender()
    
    # Test bounds validation
    print("Testing bounds validation...")
    valid, errors = recommender.validate_params("mean_reversion", "rsi_oversold", {"period": 14, "entry": 30, "exit": 50})
    print(f"Valid params: {valid}, errors: {errors}")
    
    valid, errors = recommender.validate_params("mean_reversion", "rsi_oversold", {"period": -5, "entry": 100, "exit": 50})
    print(f"Invalid params: {valid}, errors: {errors}")
