# Setup Guide

Follow these steps to set up the Forex Backtester on your local machine or VPS.

## Prerequisites
- Python 3.10+
- Git
- Access to CLIProxy (for LLM evaluations)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/faiqnau/forex-backtester.git
   cd forex-backtester
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *Note: Ensure `polars`, `joblib`, `duka`, and `httpx` are installed.*

4. **Initialize the database**:
   The `journal/database.py` will automatically create `backtest_journal.db` on first run.

## CLIProxy Configuration
Ensure CLIProxy is running at the configured endpoint (default: `http://172.27.1.12:8317`).
Update `llm/cliproxy_evaluator.py` if your endpoint or API key differs.

## Data Preparation
Download initial data using the CLI:
```bash
python3 main.py download --pair EURUSD --start 2024-01-10 --end 2024-01-11
```
