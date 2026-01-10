# Roadmap

## Phase 1: Foundation (Week 1-2)
- [x] Setup project structure
- [x] Create Dukascopy data downloader
- [x] Build Parquet storage pipeline
- [x] Test HftBacktest with sample data

## Phase 2: Strategy Library (Week 3-4)
- [x] Implement 20 base strategies (20/20 complete)
- [x] Create tuning parameter grid (5 variants each)
- [x] Build parallel strategy runner (100% success on OCI)
- [x] **XAUUSD Data Verification** (1.8GB located and verified on fight-uno)

## Phase 3: Validation Pipeline (Week 5-6)
- [x] Walk-Forward Analysis framework (Implemented)
- [x] Monte Carlo simulation (1,000 permutations, p-value, ruin probability)
| **Component** | **Tools/Files** | **Description** |
|---|---|---|
| **Metrics** | QuantStats + VectorBT | Performance analysis |
| **Analysis** | **strategy_ranker.py** | **Composite scoring & ranking** |
| **Visuals** | **visualizer.py** | **Interactive equity & trade charts** |
| **Costs** | **transaction_costs.py** | **Spread, commission, slippage modeling** |

## Phase 4: Intelligent Refinement (Week 7-8)
- [x] CLIProxy integration (Multi-model support)
- [x] Journal-based context feeding (SQLite/JSON)
- [x] **Parameter Recommender Logic** (Consensus + Bounds validation)
- [ ] **Automated Optimization Loop** (Orchestrated Sweep-Evaluate-Trial cycle)

## Phase 5: Production & Analysis (Week 9-10)
- [x] **Equity Curve & Risk Visualization** (analysis/visualizer.py)
- [x] **Weighted Strategy Ranking Dashboard** (analysis/strategy_ranker.py)
- [ ] **PDF Report Exporter**: Consolidated sweep results with embedded charts.
- [ ] **Regime Labeling**: Tag market periods (trending/ranging) for filtered analysis.
- [ ] **Multi-Currency Portfolio Engine**: Correlation matrices and capital allocation.
- [ ] **Live-Trading Bridge (Dry-Run)**: Streaming data shadow execution.


## Future Enhancements
- [ ] Advanced machine learning for regime prediction
- [ ] Real-time signal generation via Telegram/Discord
- [ ] Multi-broker adapter support
