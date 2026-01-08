"""
Multi-Pair Forex Data Downloader with Automated Resilience

Downloads all major forex pairs from Dukascopy with:
- Queue-based sequential processing
- Throttled concurrency (max 2 workers)
- AUTOMATED RESILIENCE: Hard timeouts (1h per year) via multiprocessing
- AUTOMATED RETRY: Up to 3 attempts per year
- Resume capability (skips already downloaded years)
"""

import os
import time
import json
import signal
import multiprocessing
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Dict

import polars as pl

# Try to import duka
try:
    from duka.app import app as duka_download
    from duka.core.utils import TimeFrame
except ImportError:
    print("ERROR: duka not installed. Run: pip install duka")
    exit(1)


# === CONFIGURATION ===

# Major forex pairs with their data availability start years
PAIRS = {
    "XAUUSD": 2010,   # Gold - available from 2010
    "EURUSD": 2003,   # Euro/Dollar
    "GBPUSD": 2003,   # Pound/Dollar
    "USDJPY": 2003,   # Dollar/Yen
    "USDCHF": 2003,   # Dollar/Swiss Franc
    "AUDUSD": 2003,   # Aussie/Dollar
    "USDCAD": 2003,   # Dollar/Canadian
    "NZDUSD": 2003,   # Kiwi/Dollar
}

# End year
END_YEAR = 2024

# Throttling settings
MAX_CONCURRENT_WORKERS = 2  
DELAY_BETWEEN_YEARS = 5     
DELAY_BETWEEN_PAIRS = 10    

# Automation settings
YEAR_TIMEOUT_SECONDS = 3600  # 1 hour hard timeout per year
MAX_RETRIES = 3              # Max retries if year fails or times out

# Paths
DATA_DIR = Path("data/parquet")
TEMP_BASE_DIR = Path("/tmp/duka_download")
LOG_FILE = Path("logs/download_queue.log")
STATUS_FILE = Path("data/download_status.json")


def log(msg: str):
    """Log message to file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"Logging error: {e}")


def load_status() -> Dict:
    """Load persistent status from JSON."""
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_status(status: Dict):
    """Save persistent status to JSON."""
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        log(f"Error saving status: {e}")


def get_downloaded_years(pair: str) -> List[int]:
    """Get list of already downloaded years for a pair."""
    pair_dir = DATA_DIR / pair
    if not pair_dir.exists():
        return []
    
    years = []
    for f in pair_dir.iterdir():
        if f.suffix == ".parquet":
            try:
                # Basic check: is file reasonable size?
                if f.stat().st_size > 1000 * 1024:  # > 1MB
                    years.append(int(f.stem))
            except:
                pass
    return sorted(years)


def _duka_worker(pair: str, start: date, end: date, threads: int, output_dir: str):
    """Worker function for multiprocessing."""
    try:
        duka_download(
            [pair],
            start,
            end,
            threads,
            TimeFrame.TICK,
            output_dir,
            True  # header
        )
    except Exception as e:
        print(f"Worker Error: {e}")
        os._exit(1)


def download_year_with_retry(pair: str, year: int) -> bool:
    """Download a year with timeout and retry logic."""
    output_dir = DATA_DIR / pair
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{year}.parquet"
    
    # Quick skip check
    if output_file.exists() and output_file.stat().st_size > 1000 * 1024:
        log(f"SKIP {pair}/{year} - already exists")
        return True

    status = load_status()
    key = f"{pair}_{year}"
    attemp_count = status.get(key, {}).get("retries", 0)

    if attemp_count >= MAX_RETRIES:
        log(f"SKIP {pair}/{year} - Max retries reached ({attemp_count})")
        return False

    for attempt in range(attemp_count + 1, MAX_RETRIES + 1):
        log(f"START {pair}/{year} (Attempt {attempt}/{MAX_RETRIES})")
        
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        if end > date.today(): end = date.today() - timedelta(days=1)
        if start > date.today(): return True

        temp_dir = TEMP_BASE_DIR / pair
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Run in separate process for hard timeout
        p = multiprocessing.Process(
            target=_duka_worker, 
            args=(pair, start, end, MAX_CONCURRENT_WORKERS, str(temp_dir))
        )
        p.start()
        p.join(YEAR_TIMEOUT_SECONDS)

        success = False
        if p.is_alive():
            log(f"TIMEOUT {pair}/{year} after {YEAR_TIMEOUT_SECONDS}s. Killing worker...")
            p.terminate()
            p.join()
        elif p.exitcode == 0:
            # Check if CSV was produced
            csvs = list(temp_dir.glob(f'{pair}*.csv'))
            if csvs:
                success = _process_csv(pair, year, csvs, output_file)
            else:
                log(f"WARN {pair}/{year} - Worker finished but no CSV produced")
        else:
            log(f"ERROR {pair}/{year} - Worker failed with exit code {p.exitcode}")

        if success:
            # Reset retry count on success
            status[key] = {"retries": 0, "status": "done", "last_attempt": datetime.now().isoformat()}
            save_status(status)
            return True
        else:
            status[key] = {"retries": attempt, "status": "failed", "last_attempt": datetime.now().isoformat()}
            save_status(status)
            if attempt < MAX_RETRIES:
                log(f"Retrying in 10s...")
                time.sleep(10)
    
    return False


def _process_csv(pair: str, year: int, csvs: List[Path], output_file: Path) -> bool:
    """Helper to convert duka CSV to standardized Parquet."""
    try:
        csv_path = sorted(csvs, key=os.path.getmtime)[-1]
        log(f"Converting {csv_path.name} to Parquet...")
        
        df = pl.read_csv(csv_path, try_parse_dates=True)
        
        # Standardize
        if 'time' in df.columns:
            df = df.rename({'time': 'timestamp'})
        
        # Core columns + mid/spread
        df = df.with_columns([
            ((pl.col('ask') + pl.col('bid')) / 2).alias('mid'),
            (pl.col('ask') - pl.col('bid')).alias('spread')
        ])
        
        df.write_parquet(output_file)
        log(f"DONE {pair}/{year} - {len(df):,} rows saved")
        
        # Cleanup
        csv_path.unlink()
        return True
    except Exception as e:
        log(f"Conversion error {pair}/{year}: {e}")
        return False


def run_download_queue():
    """Main queue processor."""
    log("=" * 60)
    log("STARTING AUTOMATED DOWNLOAD QUEUE")
    log(f"Pairs: {list(PAIRS.keys())}")
    log(f"Resilience: {YEAR_TIMEOUT_SECONDS}s timeout, {MAX_RETRIES} retries")
    log("=" * 60)
    
    for pair, start_year in PAIRS.items():
        log(f"\n>>> Processing {pair} (from {start_year}) <<<")
        
        downloaded = get_downloaded_years(pair)
        log(f"Valid Parquets: {downloaded}")
        
        for year in range(start_year, END_YEAR + 1):
            if year in downloaded:
                continue
            
            download_year_with_retry(pair, year)
            time.sleep(DELAY_BETWEEN_YEARS)
        
        time.sleep(DELAY_BETWEEN_PAIRS)
    
    log("\n" + "=" * 60)
    log("DOWNLOAD QUEUE COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    try:
        os.nice(10)
    except:
        pass
    
    # Increase recursion limit just in case for deep recursive calls if any (unlikely here)
    import sys
    sys.setrecursionlimit(2000)
    
    run_download_queue()
