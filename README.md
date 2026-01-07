# Forex Backtester

High-performance, distributed backtesting system for Forex strategy validation with LLM-assisted optimization.

## 🚀 Overview

The Forex Backtester is designed to handle massive-scale strategy sweeps across decades of tick data. It leverages a distributed architecture (MacBook + thinktank homelab + fight-uno VPS) and integrates with LLM agents (Claude, GPT, Gemini) to automatically analyze and refine trading strategies.

## ✨ Features

- **Massive Parallelization**: Run up to 100 concurrent backtests using `joblib`.
- **Tick-level Fidelity**: Accurate simulation with bid/ask spreads via HftBacktest.
- **22+ Year History**: Support for historical tick data from Dukascopy (2003-present).
- **Multi-Model AI Evaluation**: Deep strategy analysis from Claude Opus 4.5, Gemini 3 Pro, and GPT 5.2.
- **Intelligent Journaling**: Persistent SQLite memory to help LLMs learn from previous iterations.

## 📁 Project Structure

```bash
forex-backtester/
├── docs/              # Essential Documentation (Architecture, Changelog, Roadmap)
├── data/              # Historical Tick Data (Parquet format)
├── strategies/        # 17+ Strategy implementations (Trend, Reversion, Volatility)
├── engine/            # Core backtesting & parallel runner logic
├── journal/           # Result tracking and LLM context management
├── llm/               # CLIProxy integration for strategy evaluation
└── config/            # Parameter grids for automated sweeps
```

## 📖 Documentation

- **[Setup Guide](docs/setup.md)**: Installation and configuration.
- **[Architecture](docs/architecture.md)**: Deep dive into the distributed system.
- **[Strategy Library](docs/features.md)**: Detailed breakdown of implemented strategies.
- **[Testing Guide](docs/TESTING.md)**: How to run backtests and sweeps.
- **[Changelog](docs/CHANGELOG.md)**: Track progress and updates.
- **[Roadmap](docs/ROADMAP.md)**: Future plans and current status.

## 🛠️ Quick Start

1. **Setup environment** as described in [docs/setup.md](docs/setup.md).
2. **Run a parameter sweep**:
   ```bash
   python3 main.py sweep --n-jobs 4
   ```
3. **Evaluate with AI**:
   ```bash
   python3 main.py evaluate --input results/ --top 3
   ```

---

*Phase 3 and 4 of the foundation are complete. Strategic refinement is ongoing.*
