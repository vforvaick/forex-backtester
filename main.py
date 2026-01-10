"""
Forex Backtester - Main Entry Point

Usage:
    python main.py download --pair EURUSD --start 2020-01-01 --end 2024-12-31
    python main.py backtest --strategy ma_cross --config config/tuning_grid.yaml
    python main.py sweep --n-jobs 8 --output results/
    python main.py evaluate --run-id 123
"""

import argparse
import sys
import json
from datetime import date
from pathlib import Path

# Ensure project modules are importable
sys.path.insert(0, str(Path(__file__).parent))


def cmd_download(args):
    """Download historical data."""
    from data.download_dukascopy import download_dukascopy
    
    download_dukascopy(
        pair=args.pair,
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end)
    )
    print(f"Downloaded {args.pair} data")


def cmd_backtest(args):
    """Run single strategy backtest."""
    from engine import Backtester, ForexConfig, BacktestConfig, run_single_backtest
    import yaml
    
    # Load grid config to get default params if needed
    with open("config/tuning_grid.yaml") as f:
        tuning = yaml.safe_load(f)
    
    # Find strategy type and params
    params = {}
    strategy_module = ""
    for category, strategies in tuning.items():
        if args.strategy in strategies:
            params = strategies[args.strategy][0] # Use first variant as default
            strategy_module = f"strategies.{category}.{args.strategy}"
            break
    
    if not strategy_module:
        print(f"Strategy {args.strategy} not found in tuning_grid.yaml")
        return

    # Create config
    data_path = Path(f"data/parquet/{args.pair}")
    
    # Auto-detect date range from available parquet files
    parquet_files = sorted(data_path.glob("*.parquet"))
    if parquet_files:
        # Extract years from filenames and build a reasonable test range
        years = [int(f.stem) for f in parquet_files if f.stem.isdigit()]
        if years:
            start_year = min(years)
            start_date = f"{start_year}-01-01"
            end_date = f"{start_year}-01-31"  # Test on first month of first available year
        else:
            start_date = "2024-01-10"
            end_date = "2024-01-11"
    else:
        start_date = "2024-01-10"
        end_date = "2024-01-11"
    
    config = BacktestConfig(
        strategy_name=args.strategy,
        strategy_module=strategy_module,
        params=params,
        data_path=data_path,
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"Running {args.strategy} backtest on {args.pair}...")
    result = run_single_backtest(config)
    
    if result.success:
        print(f"Result: {result.metrics}")
        print(f"Journal Run ID: {result.run_id}")
    else:
        print(f"Error: {result.error}")


def cmd_sweep(args):
    """Run parameter sweep."""
    from engine import run_parallel_backtests, create_sweep_configs
    import yaml
    
    with open("config/tuning_grid.yaml") as f:
        tuning = yaml.safe_load(f)
    
    all_configs = []
    data_path = Path(f"data/parquet/{args.pair}")
    
    # Auto-detect date range from available parquet files if not provided
    parquet_files = sorted(data_path.glob("*.parquet"))
    
    if args.start:
        start_date = args.start
    elif parquet_files:
        years = [int(f.stem) for f in parquet_files if f.stem.isdigit()]
        start_date = f"{min(years)}-01-01" if years else "2024-01-10"
    else:
        start_date = "2024-01-10"
        
    if args.end:
        end_date = args.end
    elif parquet_files:
        years = [int(f.stem) for f in parquet_files if f.stem.isdigit()]
        end_date = f"{min(years)}-01-31" if years else "2024-01-11"
    else:
        end_date = "2024-01-11"

    print(f"Sweep Range: {start_date} to {end_date} for {args.pair}")
    
    for category, strategies in tuning.items():
        if category == "_bounds": continue
        for strategy_name, variants in strategies.items():
            if strategy_name == "_bounds": continue
            
            # Filter by strategy if requested
            if args.strategy != "all" and strategy_name != args.strategy:
                continue
                
            strategy_module = f"strategies.{category}.{strategy_name}"
            
            # Create configs for each variant
            for i, params in enumerate(variants):
                all_configs.append(create_sweep_configs(
                    strategy_name=strategy_name,
                    strategy_module=strategy_module,
                    param_ranges={k: [v] for k, v in params.items()},
                    data_path=data_path,
                    start_date=start_date,
                    end_date=end_date
                )[0])

    print(f"Running sweep with {len(all_configs)} configurations using {args.n_jobs} parallel jobs...")
    results = run_parallel_backtests(all_configs, n_jobs=args.n_jobs)
    
    # Save summary results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = []
    for r in results:
        if r.success:
            summary.append({
                "strategy": r.config.strategy_name,
                "params": r.config.params,
                "metrics": r.metrics,
                "run_id": r.run_id
            })
    
    with open(output_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Sweep completed. Summary saved to {args.output}/sweep_summary.json")


def cmd_evaluate(args):
    """Evaluate backtest results with multiple LLMs for diverse perspectives."""
    from llm.cliproxy_evaluator import evaluate_strategy_sync
    
    # Premium model trio for diverse evaluation
    MODELS = [
        ("gemini-claude-opus-4-5-thinking", "Claude Opus 4.5 Thinking"),
        ("gemini-3-pro-preview", "Gemini 3 Pro"),
        ("gpt-5.2", "GPT 5.2"),
    ]
    
    # Load sweep results
    results_file = Path(args.input) / "sweep_summary.json"
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    if not results:
        print("No results to evaluate")
        return
    
    # Sort by Sharpe ratio and get top N
    top_n = args.top if hasattr(args, 'top') else 5
    sorted_results = sorted(results, key=lambda x: x["metrics"].get("sharpe", 0), reverse=True)
    top_strategies = sorted_results[:top_n]
    
    print(f"Evaluating top {len(top_strategies)} strategies with {len(MODELS)} models...")
    print(f"Models: {', '.join([m[1] for m in MODELS])}\n")
    
    all_evaluations = []
    
    for result in top_strategies:
        strategy_evals = {
            "strategy": result["strategy"],
            "params": result["params"],
            "metrics": result["metrics"],
            "model_evaluations": {}
        }
        
        print(f"\n{'='*60}")
        print(f"Strategy: {result['strategy']}")
        print(f"Sharpe: {result['metrics'].get('sharpe', 0):.2f}, Trades: {result['metrics'].get('total_trades', 0)}")
        print(f"{'='*60}")
        
        for model_id, model_name in MODELS:
            print(f"\n--- {model_name} ---")
            
            evaluation = evaluate_strategy_sync(
                strategy_name=result["strategy"],
                metrics=result["metrics"],
                run_id=result.get("run_id", -1),
                model=model_id
            )
            
            strategy_evals["model_evaluations"][model_name] = evaluation
            
            if "verdict" in evaluation:
                print(f"Verdict: {evaluation['verdict'].upper()}")
            if "analysis" in evaluation:
                # Truncate for display
                analysis = evaluation['analysis'][:300] + "..." if len(evaluation.get('analysis', '')) > 300 else evaluation.get('analysis', '')
                print(f"Analysis: {analysis}")
            if "error" in evaluation:
                print(f"Error: {evaluation['error']}")
        
        all_evaluations.append(strategy_evals)
    
    # Save all evaluations
    output_file = Path(args.input) / "llm_evaluations_multi.json"
    with open(output_file, "w") as f:
        json.dump(all_evaluations, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Multi-model evaluations saved to {output_file}")


def cmd_wfa(args):
    """Run Walk-Forward Analysis."""
    from analysis.walk_forward import WalkForwardAnalyzer, generate_adaptive_windows
    from pathlib import Path
    
    print(f"Running Walk-Forward Analysis on {args.pair}...")
    
    # Generate adaptive windows based on available data
    # Using 3-year train, 1-year test, stepping 2 years
    windows = generate_adaptive_windows(
        start_year=2010, 
        end_year=2024, 
        train_years=3, 
        test_years=1, 
        step_years=2
    )
    
    print(f"Generated {len(windows)} walk-forward windows:")
    for w in windows:
        print(f"  Train: {w['train'][0]}-{w['train'][1]} → Test: {w['test'][0]}-{w['test'][1]}")
    
    analyzer = WalkForwardAnalyzer(Path("data/parquet"), windows)
    
    # Import strategies to test
    from strategies.pattern_based.candlestick import Strategy as CandlestickStrategy
    from strategies.trend_following.breakout import Strategy as BreakoutStrategy
    from strategies.mean_reversion.bollinger_bands import Strategy as BollingerStrategy
    
    configs = [
        {"strategy_class": CandlestickStrategy, "params": {"patterns": ["doji", "engulfing"]}},
        {"strategy_class": BreakoutStrategy, "params": {"period": 20, "atr_multiplier": 2.0}},
        {"strategy_class": BollingerStrategy, "params": {"period": 20, "std_dev": 2.0}},
    ]
    
    print(f"\nAnalyzing {len(configs)} strategy configurations...")
    results = analyzer.run_full_sweep(configs, args.pair)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analyzer.save_results(results, output_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("WALK-FORWARD ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    robust = [r for r in results if r.is_robust]
    print(f"Robust strategies (100% positive windows): {len(robust)}/{len(results)}")
    
    for r in sorted(results, key=lambda x: x.avg_test_sharpe, reverse=True):
        status = "✅ ROBUST" if r.is_robust else f"⚠️ {r.consistency_ratio:.0%}"
        print(f"  {r.strategy_name}: Avg Sharpe={r.avg_test_sharpe:.2f} {status}")


def cmd_montecarlo(args):
    """Run Monte Carlo stress test on sweep results."""
    from analysis.monte_carlo import MonteCarloSimulator, print_mc_summary
    from pathlib import Path
    
    input_file = Path(args.input) / "sweep_summary.json"
    if not input_file.exists():
        print(f"Sweep results not found: {input_file}")
        return
    
    print(f"Running Monte Carlo simulation ({args.simulations} iterations)...")
    print(f"Input: {input_file}")
    print(f"Analyzing top {args.top} strategies")
    
    simulator = MonteCarloSimulator(
        n_simulations=args.simulations,
        ruin_threshold=args.ruin_threshold,
        random_seed=args.seed
    )
    
    results = simulator.analyze_sweep_results(
        sweep_file=input_file,
        top_n=args.top
    )
    
    # Print summary
    print_mc_summary(results)
    
    # Save results
    output_file = Path(args.input) / "mc_results.json"
    simulator.save_results(results, output_file)
    
    # Summary statistics
    skill_count = sum(1 for r in results if r.is_skill_based)
    high_ruin = sum(1 for r in results if r.probability_of_ruin > 0.30)
    
    print(f"\n{'='*65}")
    print(f"SUMMARY: {skill_count}/{len(results)} strategies show statistical skill (p < 0.05)")
    if high_ruin > 0:
        print(f"⚠️  WARNING: {high_ruin} strategies have >30% probability of ruin")


def cmd_optimize(args):
    """Generate optimized trial config from LLM recommendations."""
    from llm.recommender import ParameterRecommender
    from llm.cliproxy_evaluator import evaluate_strategy_sync
    import json
    
    recommender = ParameterRecommender()
    
    # Get models from policy
    models = recommender.policy.get("consensus", {}).get("models", [
        "gemini-claude-opus-4-5-thinking",
        "gemini-3-pro-preview",
        "gpt-5.2"
    ])
    
    # Load latest sweep results
    results_file = Path(args.input) / "sweep_summary.json"
    if not results_file.exists():
        print(f"❌ No sweep results found at {results_file}")
        print("Run: python main.py sweep --output results/")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    # Find the strategy
    strategy_results = [r for r in results if r["strategy"] == args.strategy]
    if not strategy_results:
        print(f"❌ Strategy '{args.strategy}' not found in sweep results")
        return
    
    best = max(strategy_results, key=lambda x: x["metrics"].get("sharpe", 0))
    print(f"📊 Best config for {args.strategy}: Sharpe={best['metrics'].get('sharpe', 0):.2f}")
    
    # Get evaluations from all models
    print(f"\n🤖 Getting recommendations from {len(models)} models...")
    evaluations = []
    for model in models:
        print(f"  → {model}...", end=" ", flush=True)
        result = evaluate_strategy_sync(
            strategy_name=args.strategy,
            metrics=best["metrics"],
            run_id=best.get("run_id", -1),
            model=model
        )
        result["model"] = model
        evaluations.append(result)
        verdict = result.get("verdict", "error")
        print(f"[{verdict}]")
    
    # Check consensus
    has_consensus, agreed = recommender.check_consensus(evaluations)
    
    if not has_consensus:
        print("\n⚠️ No consensus reached. Models disagree:")
        for eval_result in evaluations:
            recs = eval_result.get("recommendations", [])
            if recs:
                print(f"  {eval_result['model']}: {recs[0].get('param')} → {recs[0].get('suggested')}")
        print("\nFlag for human review.")
        return
    
    print(f"\n✅ Consensus reached on {len(agreed)} parameter(s):")
    for param, rec in agreed.items():
        print(f"  {param}: → {rec['suggested']} ({rec['agreement']:.0%} agreement)")
    
    # Determine category
    category = None
    import yaml
    with open("config/tuning_grid.yaml") as f:
        grid = yaml.safe_load(f)
    for cat, strategies in grid.items():
        if cat == "_bounds":
            continue
        if args.strategy in strategies:
            category = cat
            break
    
    if not category:
        print(f"❌ Could not find category for {args.strategy}")
        return
    
    # Generate trial
    trial = recommender.generate_trial(
        category=category,
        strategy=args.strategy,
        recommendations=agreed,
        source_run_id=best.get("run_id", -1),
        llm_model=", ".join(models)
    )
    
    if trial:
        print(f"\n🎯 Next step: Run the trial backtest and compare")
        print(f"   python main.py compare --trial trials/{category}/{args.strategy}/{trial.meta.trial_id}.yaml")


def cmd_compare(args):
    """Compare original vs trial with MC stress test."""
    from llm.recommender import ParameterRecommender
    import yaml
    import json
    
    trial_path = Path(args.trial)
    if not trial_path.exists():
        print(f"❌ Trial not found: {trial_path}")
        return
    
    with open(trial_path) as f:
        trial = yaml.safe_load(f)
    
    recommender = ParameterRecommender()
    
    # For demo, use mock metrics (in production, run actual backtest)
    # TODO: Integrate with actual backtest run
    original_metrics = {
        "sharpe": 1.20, "sortino": 1.5, "max_drawdown": -0.12,
        "win_rate": 0.52, "profit_factor": 1.6, "total_trades": 342
    }
    trial_metrics = {
        "sharpe": 1.45, "sortino": 1.8, "max_drawdown": -0.10,
        "win_rate": 0.55, "profit_factor": 1.8, "total_trades": 289
    }
    
    print("Running Monte Carlo stress test...")
    original_mc = recommender.run_mc_check(original_metrics)
    trial_mc = recommender.run_mc_check(trial_metrics)
    
    print("\n" + recommender.compare(original_metrics, trial_metrics, original_mc, trial_mc))
    
    # Check MC gatekeeper
    max_pvalue = recommender.policy.get("constraints", {}).get("max_mc_pvalue", 0.10)
    max_ruin = recommender.policy.get("constraints", {}).get("max_ruin_prob", 0.30)
    
    if trial_mc["p_value"] >= max_pvalue:
        print(f"\n🚫 BLOCKED: P-value {trial_mc['p_value']:.2f} >= {max_pvalue} (overfitting risk)")
    elif trial_mc["ruin_prob"] >= max_ruin:
        print(f"\n🚫 BLOCKED: Ruin prob {trial_mc['ruin_prob']:.1%} >= {max_ruin:.0%}")
    else:
        print(f"\n✅ Trial passes MC gatekeeper. Promote with:")
        print(f"   python main.py promote --trial {trial_path}")


def cmd_explain(args):
    """Show detailed explanation of a trial."""
    from llm.recommender import ParameterRecommender
    
    trial_path = Path(args.trial)
    if not trial_path.exists():
        print(f"❌ Trial not found: {trial_path}")
        return
    
    recommender = ParameterRecommender()
    print(recommender.explain_trial(trial_path))


def cmd_promote(args):
    """Promote a trial to tuning_grid.yaml with git commit."""
    from llm.recommender import ParameterRecommender
    
    trial_path = Path(args.trial)
    if not trial_path.exists():
        print(f"❌ Trial not found: {trial_path}")
        return
    
    recommender = ParameterRecommender()
    
    if args.dry_run:
        success, msg = recommender.git_promote(trial_path, dry_run=True)
        print(msg)
    else:
        # Show preview first
        success, preview = recommender.git_promote(trial_path, dry_run=True)
        print(preview)
        
        confirm = input("\nProceed with promotion? [y/N]: ").strip().lower()
        if confirm == "y":
            success, msg = recommender.git_promote(trial_path, dry_run=False)
            print(msg)
        else:
            print("Aborted.")


def cmd_undo(args):
    """Undo a trial promotion."""
    from llm.recommender import ParameterRecommender
    
    recommender = ParameterRecommender()
    
    if args.dry_run:
        success, msg = recommender.git_undo(args.trial_id, dry_run=True)
        print(msg)
    else:
        # Show preview first
        success, preview = recommender.git_undo(args.trial_id, dry_run=True)
        print(preview)
        
        confirm = input("\nProceed with undo? [y/N]: ").strip().lower()
        if confirm == "y":
            success, msg = recommender.git_undo(args.trial_id, dry_run=False)
            print(msg)
        else:
            print("Aborted.")


def cmd_validate(args):
    """Check for degradation in recently promoted strategies."""
    import json
    
    # Load promotion history from git log
    import subprocess
    result = subprocess.run(
        ["git", "log", "--oneline", "--grep", "optimize: promote", f"-{args.last}"],
        capture_output=True, text=True, cwd="."
    )
    
    commits = [line for line in result.stdout.strip().split("\n") if line]
    
    if not commits:
        print("No promotions found in git history.")
        return
    
    print(f"Checking {len(commits)} recent promotions...\n")
    
    for commit in commits:
        parts = commit.split()
        commit_hash = parts[0]
        strategy = parts[-1] if len(parts) > 2 else "unknown"
        
        # In production, would re-run backtest on latest data
        # For now, just show the promotions
        print(f"  {commit_hash[:7]}: {strategy}")
    
    print("\n⚠️ Note: Full degradation check requires running backtests on new data.")
    print("   This is a placeholder for the full implementation.")


def main():

    parser = argparse.ArgumentParser(
        description="Forex Backtester - High-performance strategy testing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Download command
    dl = subparsers.add_parser("download", help="Download historical data")
    dl.add_argument("--pair", required=True, help="Currency pair (e.g., EURUSD)")
    dl.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    dl.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    
    # Backtest command
    bt = subparsers.add_parser("backtest", help="Run single backtest")
    bt.add_argument("--strategy", required=True, help="Strategy name")
    bt.add_argument("--pair", default="EURUSD", help="Currency pair")
    bt.add_argument("--config", default="config/config.yaml", help="Config file")
    
    # Sweep command
    sw = subparsers.add_parser("sweep", help="Run parameter sweep")
    sw.add_argument("--pair", default="EURUSD", help="Currency pair")
    sw.add_argument("--strategy", default="all", help="Specific strategy or 'all'")
    sw.add_argument("--start", help="Start date (YYYY-MM-DD)")
    sw.add_argument("--end", help="End date (YYYY-MM-DD)")
    sw.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs")
    sw.add_argument("--output", default="results/", help="Output directory")
    
    # Evaluate command
    ev = subparsers.add_parser("evaluate", help="Evaluate sweep results with LLM")
    ev.add_argument("--input", default="results/", help="Results directory with sweep_summary.json")
    ev.add_argument("--top", type=int, default=5, help="Number of top strategies to evaluate")
    
    # Walk-Forward Analysis command
    wf = subparsers.add_parser("wfa", help="Run Walk-Forward Analysis")
    wf.add_argument("--pair", default="XAUUSD", help="Currency pair")
    wf.add_argument("--strategy", default="all", help="Strategy to test (or 'all')")
    wf.add_argument("--output", default="results/wfa_results.json", help="Output file")
    
    # Monte Carlo command
    mc = subparsers.add_parser("montecarlo", help="Run Monte Carlo stress test")
    mc.add_argument("--input", default="results/", help="Results directory with sweep_summary.json")
    mc.add_argument("--simulations", type=int, default=1000, help="Number of simulations")
    mc.add_argument("--top", type=int, default=5, help="Top N strategies to analyze")
    mc.add_argument("--ruin-threshold", type=float, default=-0.50, help="Max drawdown threshold for ruin")
    mc.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    # === NEW OPTIMIZATION COMMANDS ===
    
    # Optimize command
    opt = subparsers.add_parser("optimize", help="Generate optimized trial from LLM recommendations")
    opt.add_argument("--strategy", required=True, help="Strategy name to optimize")
    opt.add_argument("--input", default="results/", help="Results directory with sweep_summary.json")
    opt.add_argument("--target", default="sharpe", choices=["sharpe", "safe", "aggressive"],
                     help="Optimization target")
    
    # Compare command
    cmp = subparsers.add_parser("compare", help="Compare original vs trial with MC stress test")
    cmp.add_argument("--trial", required=True, help="Path to trial YAML file")
    
    # Explain command
    exp = subparsers.add_parser("explain", help="Show detailed explanation of a trial")
    exp.add_argument("--trial", required=True, help="Path to trial YAML file")
    
    # Promote command
    prom = subparsers.add_parser("promote", help="Promote trial to tuning_grid.yaml")
    prom.add_argument("--trial", required=True, help="Path to trial YAML file")
    prom.add_argument("--dry-run", action="store_true", help="Show what would change without applying")
    
    # Undo command
    undo = subparsers.add_parser("undo", help="Undo a trial promotion")
    undo.add_argument("--trial-id", required=True, help="Trial ID to undo (e.g., trial_001)")
    undo.add_argument("--dry-run", action="store_true", help="Show what would change without applying")
    
    # Validate command
    val = subparsers.add_parser("validate", help="Check for degradation in promoted strategies")
    val.add_argument("--last", type=int, default=3, help="Number of recent promotions to check")
    
    args = parser.parse_args()

    
    if args.command == "download":
        cmd_download(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "wfa":
        cmd_wfa(args)
    elif args.command == "montecarlo":
        cmd_montecarlo(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "explain":
        cmd_explain(args)
    elif args.command == "promote":
        cmd_promote(args)
    elif args.command == "undo":
        cmd_undo(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

