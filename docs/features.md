# Features & Strategy Library

The Forex Backtester implements a comprehensive library of 20 trading strategies across 5 major categories. Each strategy is designed to be self-contained and compatible with parallel execution.

## Strategy Categories

### 1. Trend Following
Designed to identify and capitalize on sustained market movements.
- **Moving Average Crossover**: SMA/EMA crossovers with configurable periods.
- **Breakout**: Price breakouts based on Donchian Channels or ATR-based levels.
- **Momentum**: Signal generation using MACD, RSI, and Stochastic oscillators.
- **ADX Trend**: Trend strength filtering using the Average Directional Index.

### 2. Mean Reversion
Capitalizes on price returning to a mean or average value.
- **Bollinger Bands**: Mean reversion signals from band touches and squeezes.
- **RSI Oversold/Overbought**: Trading extreme momentum readings.
- **Price Channel**: Reversions from Keltner Channels or Envelopes.
- **Statistical Arbitrage**: Z-score based deviations from historical averages.

### 3. Volatility
Trades based on changes in market volatility or range expansion.
- **ATR Trailing Stop**: Volatility-adjusted stop-loss management.
- **Volatility Breakout**: Squeeze detection and breakout trading.
- **Range Trading**: Identifying and trading within price ranges.

### 4. Pattern Based
Identifies specific price formations and candlestick patterns.
- **Candlestick Patterns**: Automated detection of Hammer, Engulfing, Doji, etc.
- **Price Action**: Pin bars and inside bar patterns with confirmation.
- **Harmonic Patterns**: Fibonacci-based Gartley/Bat pattern detection.

### 5. Hybrid
Combines multiple signals or timeframes for higher confidence.
- **Multi-Timeframe**: Confirmation from higher timeframe trends.
- **Regime Switching**: Dynamically switching between trend and range logic.
- **Ensemble**: Voting-based combination of multiple strategy signals.

## Core Features
- **Massive Parallelization**: Support for up to 100 concurrent backtests using `joblib`.
- **Tick-Level Accuracy**: Integration with HftBacktest for high-fidelity simulations.
- **LLM Evaluation**: Automated strategy analysis using Claude, GPT-5, and Gemini.
- **Memory-Efficient Data**: Polars and Parquet pipeline for handling 20+ years of tick data.
- **Backtesting Journal**: Persistence of results and AI insights in SQLite.
- **Monte Carlo Simulation**: Stress testing framework that shuffles trade sequences 1,000+ times to calculate p-values (luck vs skill) and probability of ruin.
- **Walk-Forward Analysis**: Dynamic window validation (3yr train / 1yr test) to detect overfitting across market regimes.

## Automated Parameter Optimization
A closed-loop system for LLM-driven strategy refinement with human oversight.

- **Multi-Model Consensus**: Requires 2/3 agreement from Claude, GPT, Gemini before generating trials.
- **Parameter Bounds**: Validates LLM suggestions against `_bounds` in `tuning_grid.yaml`.
- **Monte Carlo Gatekeeper**: Blocks trials with p-value ≥0.10 or ruin probability ≥30%.
- **Optimization Policy**: Configurable weights, constraints, and consensus thresholds in `optimization_policy.yaml`.
- **Git-based Reversibility**: Promotions via git commit, undo via git revert.

### CLI Commands
```bash
python main.py optimize --strategy rsi_oversold     # Generate trial from LLM
python main.py compare --trial trials/.../trial.yaml  # A/B with MC metrics
python main.py explain --trial trials/.../trial.yaml  # Show LLM reasoning
python main.py promote --trial trials/.../trial.yaml  # Git commit
python main.py undo --trial-id trial_001             # Git revert
python main.py validate --last 3                     # Degradation check
```
