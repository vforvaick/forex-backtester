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
- [x] Result aggregation & ranking (Parquet sweep summary)
- [x] **Realistic Transaction Costs** (Spread, Commission, Slippage modeling)

## Phase 4: LLM Integration (Week 7-8)
- [x] CLIProxy integration (Multi-model support)
- [x] Journal-based context feeding (SQLite/JSON)
- [x] Automated parameter recommendation loop (Completed)


## Future Enhancements
- [ ] Live trading adapter
- [ ] Multi-currency portfolio optimization
- [ ] Real-time signal generation
