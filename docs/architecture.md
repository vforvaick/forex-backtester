# Forex Backtester Architecture

## Overview

High-performance, distributed backtesting system for Forex strategy validation with LLM-assisted optimization.

## System Architecture

```mermaid
graph TB
    subgraph DEV["MacBook M3 8GB"]
        A[Strategy Development]
        B[Quick Tests 2-week data]
    end
    
    subgraph MEDIUM["thinktank 8t/16GB"]
        C[Medium Batch 8 parallel]
        D[5-10 years validation]
    end
    
    subgraph PROD["fight-uno 4c/24GB"]
        E[Full Scale 100 backtests]
        F[22 years data]
        G[Walk-Forward + Monte Carlo]
    end
    
    subgraph LLM["fight-dos CLIProxy"]
        H[Claude Opus 4.5 Thinking]
        I[Gemini 3 Pro]
        J[GPT 5.2]
    end
    
    A --> B --> C --> D --> E --> F --> G --> H & I & J
    H & I & J -->|Diverse Evaluations| K[Ranking & Refinement]
    K -->|Refined params| A

    
    subgraph JOURNAL["SQLite Journal"]
        J[backtest_runs]
        K[llm_evaluations]
        L[insights]
    end
    
    G --> J
    H --> K
    K --> L
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Engine | HftBacktest (Rust + Python) | Tick-level backtesting |
| Data | Polars + Parquet | Memory-efficient data handling |
| Metrics | QuantStats + VectorBT | Performance analysis |
| **Costs** | **transaction_costs.py** | **Spread, commission, slippage modeling** |
| Parallelization | joblib | Distributed execution |
| Journal | SQLite | Result tracking & LLM memory |
| LLM | CLIProxy (Claude/GPT/Gemini) | Strategy evaluation |

## Data Sources

- **Primary**: Dukascopy (~22 years tick data)
- **Secondary**: HistData.com (M1 bars, extended history)
- **Format**: Parquet, partitioned by symbol/year

## Compute Resources

| Node | Specs | Role |
|------|-------|------|
| MacBook M3 | 8GB RAM | Development |
| thinktank.local | i5-8350U 8t, 16GB | Medium batch |
| fight-uno | 4c ARM, 24GB | Full scale |
| fight-dos | CLIProxy host | LLM evaluation |

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

## Validation Methodology

1. **Walk-Forward Analysis**: 6 rolling windows across 22 years
2. **Monte Carlo**: 1000 trade sequence permutations
3. **Regime Analysis**: Performance across market conditions

## References

- [Implementation Plan](file:///Users/faiqnau/.gemini/antigravity/brain/0847d21c-2aa5-4311-8d02-74ee32715964/implementation_plan.md)
