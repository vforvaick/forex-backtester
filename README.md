# Forex Backtester

High-performance, distributed backtesting system for Forex strategy validation with LLM-assisted optimization.

## Features

- **Massive Parallelization**: 100 concurrent backtests (20 strategies × 5 tunings)
- **Distributed Compute**: MacBook (dev) → thinktank (validation) → VPS (production)
- **22 Years of Data**: Tick-level accuracy via Dukascopy
- **LLM Integration**: Claude/GPT/Gemini via CLIProxy for intelligent parameter tuning
- **Backtesting Journal**: SQLite-based memory for LLM context retention

## Quick Start

```bash
# Install dependencies
pip install polars joblib httpx pyyaml quantstats

# Optional: Full HftBacktest support
pip install hftbacktest duka

# Download data
python main.py download --pair EURUSD --start 2020-01-01 --end 2024-12-31

# Run single backtest
python main.py backtest --strategy ma_cross --pair EURUSD

# Run parameter sweep
python main.py sweep --n-jobs 8

# Evaluate with LLM
python main.py evaluate --run-id 123
```

## Architecture

```
MacBook M3     →  thinktank.local  →  fight-uno VPS  →  fight-dos CLIProxy
(Development)     (8-parallel)        (Full Scale)       (LLM Evaluation)
```

## Directory Structure

```
forex-backtester/
├── docs/              # Documentation
├── data/              # Tick data (Parquet)
├── strategies/        # 20 base strategies
├── engine/            # Backtesting core
├── analysis/          # Metrics & validation
├── journal/           # Result tracking
├── llm/               # CLIProxy integration
└── config/            # Tuning parameters
```

## Documentation

- [Architecture](docs/architecture.md)
- [Changelog](docs/CHANGELOG.md)
- [Roadmap](docs/ROADMAP.md)

## Compute Nodes

| Node | Specs | Role |
|------|-------|------|
| MacBook M3 | 8GB RAM | Development |
| thinktank.local | i5-8350U 8t, 16GB | Medium batch (8 parallel) |
| fight-uno | 4c ARM, 24GB | Full scale production |
| fight-dos | CLIProxy | LLM evaluation |

## License

MIT
