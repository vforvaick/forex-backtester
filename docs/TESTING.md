# Testing Guide

This document outlines how to verify the functionality of the Forex Backtester.

## 1. Unit Tests
(Standard unit tests to be implemented in `tests/` directory)

## 2. Integration Testing

### Single Backtest
Test a single strategy configuration:
```bash
python3 main.py backtest --strategy trend_following.moving_average_crossover --params '{"fast_period": 10, "slow_period": 50}'
```

### Parallel Parameter Sweep
Run a full sweep across all strategies defined in `config/tuning_grid.yaml`:
```bash
python3 main.py sweep --n-jobs 4
```

### LLM Evaluation
Run AI-powered analysis on sweep results:
```bash
python3 main.py evaluate --input results/ --top 3
```

## 3. Verification Protocol
After making changes to strategy logic:
1. Run a 2-day sample backtest to check for `ColumnNotFoundError`.
2. Run a small parallel sweep (2-3 jobs) to verify thread safety.
3. Verify that the SQLite journal correctly records the results.
4. Run LLM evaluation to catch any "impossible" metrics (e.g., Sharpe > 10 with 0 trades).

## 4. Known Data Issues
- **Dukascopy URLs**: If downloads fail, verify that `duka` is patched to use `datafeed.dukascopy.com`.
- **ISP Blocks**: Some ISPs block Dukascopy data; use a VPS or VPN if redirects occur.
