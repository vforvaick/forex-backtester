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
    config = BacktestConfig(
        strategy_name=args.strategy,
        strategy_module=strategy_module,
        params=params,
        data_path=data_path,
        start_date="2024-01-10", # Sample range
        end_date="2024-01-11"
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
    data_path = Path(f"data/parquet/EURUSD") # Default to EURUSD for now
    
    for category, strategies in tuning.items():
        for strategy_name, variants in strategies.items():
            strategy_module = f"strategies.{category}.{strategy_name}"
            
            # Create configs for each variant
            for i, params in enumerate(variants):
                all_configs.append(create_sweep_configs(
                    strategy_name=strategy_name,
                    strategy_module=strategy_module,
                    param_ranges={k: [v] for k, v in params.items()}, # Wrap in list for grid gen
                    data_path=data_path,
                    start_date="2024-01-10",
                    end_date="2024-01-11"
                )[0]) # Since we passed single values, we get 1 config per variant

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
    sw.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs")
    sw.add_argument("--output", default="results/", help="Output directory")
    
    # Evaluate command
    ev = subparsers.add_parser("evaluate", help="Evaluate sweep results with LLM")
    ev.add_argument("--input", default="results/", help="Results directory with sweep_summary.json")
    ev.add_argument("--top", type=int, default=5, help="Number of top strategies to evaluate")
    
    args = parser.parse_args()
    
    if args.command == "download":
        cmd_download(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
