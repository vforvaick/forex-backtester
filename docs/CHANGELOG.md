# Changelog

All notable changes to forex-backtester will be documented in this file.

## [2026-01-11] - Progress Bar & Early Stop
### Added
- **Progress Bar + ETA**: `engine/parallel_runner.py` now uses `tqdm` for real-time sweep progress with ETA.
- **Early Stop Filter**: Rejects strategies mid-sweep if DD>50%, PF<0.5, or trades<10. Configurable in `optimization_policy.yaml`.
- **Early Stopped Verdict**: Journal database now accepts `early_stopped` as a valid verdict.

### Files Modified
- `engine/parallel_runner.py`
- `config/optimization_policy.yaml`
- `journal/database.py`

### Reference
- Session: eeef757d-c017-4958-bb7d-dc7eb72aeb2d

## [2026-01-11] - Visualization & Quantitative Ranking
### Added
- **Quantitative Strategy Ranker**: `analysis/strategy_ranker.py` with composite scoring (Sharpe 25%, Calmar 20%, Sortino 15%, PF 15%, DD 10%, RF 10%, WR 5%).
- **Equity Curve Visualizer**: `analysis/visualizer.py` with Plotly charts for equity curves, drawdown, and metrics dashboards.
- **LLM Context Enhancement**: `llm/cliproxy_evaluator.py` now accepts trade distribution (hour/day), outlier trades, and anomaly detection.
- **Ranking Weights Config**: `config/optimization_policy.yaml` now has `ranking_weights` section for customizable scoring.

### Files Modified
- `analysis/strategy_ranker.py` [NEW]
- `analysis/visualizer.py` [NEW]
- `analysis/__init__.py`
- `config/optimization_policy.yaml`
- `llm/cliproxy_evaluator.py`

### Reference
- Session: eeef757d-c017-4958-bb7d-dc7eb72aeb2d

## [2026-01-11] - System Run & XAUUSD Baseline
### Added
- **Synthetic Data Generator**: `scripts/generate_test_data.py` for local testing without Dukascopy dependencies.
- **Dynamic Date Detection**: `main.py` now automatically detects available parquet files and sets backtest range.
- **XAUUSD Baseline**: Successfully ran `moving_average_crossover` on 9 years of XAUUSD data (2010-2018) on `fight-uno`.
- **Phase 5 Roadmap**: Defined Production & Portfolio Scaling roadmap, including Multi-Currency Portfolio Engine and Live-Trading Bridge.

### Fixed
- **Import Error**: Fixed missing `run_single_backtest` export in `engine/__init__.py`.
- **Data Schema Compatibility**: Updated synthetic data generator to include `bid_vol`, `ask_vol`, and `mid` columns to match strategy expectations.
- **VPS Environment**: Managed Python venv and dependencies on `fight-uno` for high-memory backtesting.

### Files Modified
- `engine/__init__.py`
- `main.py`
- `scripts/generate_test_data.py` [NEW]

### Reference
- **Journal Run ID**: 391 (XAUUSD Jan 2010)
- Session: eeef757d-c017-4958-bb7d-dc7eb72aeb2d

## [Unreleased]

### Added
- Initial project structure
- Architecture documentation
- Distributed compute design (MacBook + thinktank + fight-uno + fight-dos)
- Journal system schema for LLM context retention
- **20 trading strategies across 5 categories**:
  - Trend Following: MA Crossover, Breakout, Momentum, ADX
  - Mean Reversion: Bollinger Bands, RSI Oversold, Price Channel, Statistical Arb
  - Volatility: ATR Trailing, Volatility Breakout, Range Trading
  - Pattern Based: Candlestick, Price Action, Harmonic
  - Hybrid: Multi-Timeframe, Regime Switching, Ensemble

## [2026-01-08] - Automated Parameter Recommendation Loop
### Added
- **6 new CLI commands**: `optimize`, `compare`, `explain`, `promote`, `undo`, `validate`
- **Multi-Model Consensus**: 2/3 LLM agreement required before generating trials
- **Parameter Bounds Validation**: `_bounds` in `tuning_grid.yaml` prevents LLM hallucinations
- **Monte Carlo Gatekeeper**: Blocks trials with p-value ≥0.10 or ruin prob ≥30%
- **Optimization Policy**: `config/optimization_policy.yaml` (weights, constraints, consensus threshold)
- **Git-based Reversibility**: Promotions via git commit, undo via git revert
- **Dry-run Mode**: Preview changes before promotion

### New Files
- `llm/recommender.py`: `ParameterRecommender` class with consensus, validation, trial generation
- `config/optimization_policy.yaml`: Policy config for optimization loop
- `trials/`: Strategy-scoped trial directory structure

### Reference
- Session: f39a4567-70fc-47b2-8303-9b27c3636fa5

## [2026-01-08] - Transaction Cost Model
### Added
- **Realistic Transaction Costs**: `engine/transaction_costs.py` with spread, commission, slippage modeling
  - Presets for EURUSD (0.5 pip), XAUUSD (2.5 pip), GBPUSD (0.8 pip)
- **Centralized Metrics**: `strategies/base.py` with `calculate_metrics_with_costs()` used by all 17 strategies
- Cost breakdown in metrics: spread_cost_pips, commission_usd, cost_per_trade_pips

### Impact
- High-frequency strategies: 50-80% Sharpe reduction (expected behavior)
- Low-frequency strategies: 10-30% reduction

### Reference
- Session: c1beaca0-2557-42ac-bbd6-ca6a5e60e5dc

## [2026-01-08] - Strategy Library 20/20 Complete
### Added
- **Price Action Strategy Enhanced**: Proper Pin Bar/Inside Bar detection with body/wick ratio thresholds, lookahead fix (removed `shift(-1)`), confirmation mechanism for breakouts, full 8-metric calculation with **transaction costs integration**.
- **Harmonic Patterns Strategy Enhanced**: XABCD swing point detection, Gartley/Bat pattern recognition using Fibonacci ratios (0.618, 0.786, 0.886), momentum-based confirmation signals, complete metrics with **transaction costs integration**.
- **Regime Switching Strategy Enhanced**: Proper ADX calculation with directional movement, ATR-based volatility regime detection (percentile-based), adaptive strategy switching (trend-following in trending markets, mean-reversion in ranging low-vol), stayed-out logic for choppy markets, and **transaction costs integration**.

## [2026-01-08] - Phase 3 & 4 Completion
### Added
- **100% Parallel Backtesting Success**: All 85 strategy variants (17 base strategies × 5 variants) now run in parallel on `fight-uno` VPS.
- **Dukascopy 2024 Data**: Downloaded EURUSD tick data (Jan 10-11, 2024) and converted to Parquet on `fight-uno`.
- **CLIProxy Multi-Model Evaluation**: Integrated `gemini-claude-opus-4-5-thinking`, `gemini-3-pro-preview`, and `gpt-5.2` for strategy analysis.
- **Self-Contained Strategies**: Refactored `_generate_signals` for all 17 strategies to ensure parallel execution compatibility with automatic indicator calculation.
- **Automated XAUUSD Download**: Background script `download_xauusd.py` for full historical (2010-2024) tick data acquisition.
- **Walk-Forward Analysis Framework**: Rolling window validation with adaptive window generation (3yr train / 1yr test / 2yr step). Detects overfitting by testing on out-of-sample data.
- **Monte Carlo Simulation**: Stress testing framework with 1,000 trade sequence permutations, p-value calculation (luck vs skill), probability of ruin estimation, and 95% confidence intervals for Sharpe ratio. CLI: `python main.py montecarlo --input results/`.

### Fixed
- Patched `duka` library URL from `www.dukascopy.com` to `datafeed.dukascopy.com` to bypass 301 redirects.
- Resolved `ColumnNotFoundError` in parallel execution by ensuring lazy Polars expressions are evaluated on dataframes with materialized indicators.
- Fixed Polars method name consistency (`cum_prod` vs `cumprod`).
- Updated `Backtester` to handle strategy functions returning full metrics dictionaries.
- **Candlestick Strategy Bug** (2026-01-08): Fixed 0-trades issue. Root cause was signal expression not being evaluated to Series, missing `mid` column calculation, and incorrect Polars filter syntax. Now produces 7,427 trades with Sharpe 33.78, Win Rate 5.41%, PF 1.62.

### Discovered Issues
- ~~**Candlestick Strategy Bug**: Reported Sharpe 84.89 with 0 trades but positive total return (4.7%). Confirmed by all 3 LLM models as a metrics calculation or signal generation flaw.~~ **FIXED**
- **ISP Blocking**: Verified that MyRepublic (thinktank) blocks/redirects Dukascopy datafeed. Switched primary data acquisition to OCI (fight-uno).

### Files Modified
- `main.py`: Implemented `evaluate` command with multi-model support. Added `montecarlo` command for Monte Carlo stress testing.
- `engine/parallel_runner.py`: Updated strategy function wrapper.
- `engine/backtester.py`: Modified `_run_simple_backtest` to accept metrics dictionaries.
- `llm/cliproxy_evaluator.py`: Configured internal VCN endpoint for CLIProxy.
- `strategies/*`: Refactored all 17 strategy files.
- `analysis/monte_carlo.py`: New Monte Carlo simulation module with `MonteCarloSimulator` class.
- `analysis/__init__.py`: Added Monte Carlo exports.

### Reference
- Session: c5730ad8-bf78-4672-9212-b913b68dea48

