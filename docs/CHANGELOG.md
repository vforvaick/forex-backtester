# Changelog

All notable changes to forex-backtester will be documented in this file.

## [Unreleased]

### Added
- Initial project structure
- Architecture documentation
- Distributed compute design (MacBook + thinktank + fight-uno + fight-dos)
- Journal system schema for LLM context retention
- **17 trading strategies across 5 categories**:
  - Trend Following: MA Crossover, Breakout, Momentum, ADX
  - Mean Reversion: Bollinger Bands, RSI Oversold, Price Channel, Statistical Arb
  - Volatility: ATR Trailing, Volatility Breakout, Range Trading
  - Pattern Based: Candlestick, Price Action, Harmonic
  - Hybrid: Multi-Timeframe, Regime Switching, Ensemble

## [2026-01-08] - Phase 3 & 4 Completion
### Added
- **100% Parallel Backtesting Success**: All 85 strategy variants (17 base strategies × 5 variants) now run in parallel on `fight-uno` VPS.
- **Dukascopy 2024 Data**: Downloaded EURUSD tick data (Jan 10-11, 2024) and converted to Parquet on `fight-uno`.
- **CLIProxy Multi-Model Evaluation**: Integrated `gemini-claude-opus-4-5-thinking`, `gemini-3-pro-preview`, and `gpt-5.2` for strategy analysis.
- **Self-Contained Strategies**: Refactored `_generate_signals` for all 17 strategies to ensure parallel execution compatibility with automatic indicator calculation.
- **Automated XAUUSD Download**: Background script `download_xauusd.py` for full historical (2010-2024) tick data acquisition.

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
- `main.py`: Implemented `evaluate` command with multi-model support.
- `engine/parallel_runner.py`: Updated strategy function wrapper.
- `engine/backtester.py`: Modified `_run_simple_backtest` to accept metrics dictionaries.
- `llm/cliproxy_evaluator.py`: Configured internal VCN endpoint for CLIProxy.
- `strategies/*`: Refactored all 17 strategy files.

### Reference
- Session: 0847d21c-2aa5-4311-8d02-74ee32715964

