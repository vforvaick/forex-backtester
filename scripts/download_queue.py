"""
Multi-Pair Forex Data Downloader with Throttling

Downloads all major forex pairs from Dukascopy with:
- Queue-based sequential processing
- Throttled concurrency (max 2 workers)
- Delay between requests to prevent rate limiting
- Resume capability (skips already downloaded years)
"""

import os
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

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
MAX_CONCURRENT_WORKERS = 2  # Reduced from default 10
DELAY_BETWEEN_YEARS = 5     # Seconds between year downloads
DELAY_BETWEEN_PAIRS = 10    # Seconds between pair switches

# Paths
DATA_DIR = Path("data/parquet")
TEMP_DIR = Path("data/temp")
LOG_FILE = Path("logs/download_queue.log")


def log(msg: str):
    """Log message to file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_downloaded_years(pair: str) -> List[int]:
    """Get list of already downloaded years for a pair."""
    pair_dir = DATA_DIR / pair
    if not pair_dir.exists():
        return []
    
    years = []
    for f in pair_dir.iterdir():
        if f.suffix == ".parquet":
            try:
                year = int(f.stem)
                # Verify file is not empty/corrupt
                df = pl.read_parquet(f)
                if len(df) > 1000:  # Minimum viable data
                    years.append(year)
            except:
                pass
    return sorted(years)


def download_year(pair: str, year: int) -> bool:
    """Download a single year of tick data for a pair. Based on working download_xauusd.py."""
    from datetime import date, timedelta
    import os
    
    output_dir = DATA_DIR / pair
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{year}.parquet"
    
    # Skip if already exists and valid
    if output_file.exists():
        try:
            df = pl.read_parquet(output_file)
            if len(df) > 1000:
                log(f"SKIP {pair}/{year} - already downloaded ({len(df):,} rows)")
                return True
        except:
            log(f"WARN {pair}/{year} - corrupt file, re-downloading")
            output_file.unlink()
    
    log(f"START {pair}/{year}")
    
    # Use date (not datetime) - same as working script
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    
    # Don't download future dates
    if end > date.today():
        end = date.today() - timedelta(days=1)
    
    if start > date.today():
        log(f"SKIP {pair}/{year} - future date")
        return True
    
    # Use temp directory
    temp_dir = Path('/tmp/duka_download') / pair
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download using duka - EXACT same call as working script
        duka_download(
            [pair],
            start,
            end,
            MAX_CONCURRENT_WORKERS,  # threads (reduced from default)
            TimeFrame.TICK,
            str(temp_dir),
            True  # header
        )
        
        # Find and convert CSV
        csvs = list(temp_dir.glob(f'{pair}*.csv'))
        if csvs:
            csv_path = sorted(csvs, key=os.path.getmtime)[-1]
            log(f"Converting {csv_path.name} to Parquet...")
            
            df = pl.read_csv(csv_path, try_parse_dates=True)
            
            # Standardize column names
            if 'time' in df.columns:
                df = df.rename({'time': 'timestamp'})
            
            # Add mid price and spread
            df = df.with_columns([
                ((pl.col('ask') + pl.col('bid')) / 2).alias('mid'),
                (pl.col('ask') - pl.col('bid')).alias('spread')
            ])
            
            df.write_parquet(output_file)
            log(f"DONE {pair}/{year} - {len(df):,} rows saved")
            
            # Cleanup
            csv_path.unlink()
            return True
        else:
            log(f"WARN {pair}/{year} - no CSV found")
            return False
            
    except Exception as e:
        log(f"ERROR {pair}/{year} - {e}")
        return False


def run_download_queue():
    """Main queue processor."""
    log("=" * 60)
    log("STARTING MULTI-PAIR DOWNLOAD QUEUE")
    log(f"Pairs: {list(PAIRS.keys())}")
    log(f"End Year: {END_YEAR}")
    log(f"Throttle: {MAX_CONCURRENT_WORKERS} workers, {DELAY_BETWEEN_YEARS}s delay")
    log("=" * 60)
    
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    
    for pair, start_year in PAIRS.items():
        log(f"\n>>> Processing {pair} (from {start_year}) <<<")
        
        downloaded_years = get_downloaded_years(pair)
        log(f"Already have: {downloaded_years}")
        
        for year in range(start_year, END_YEAR + 1):
            if year in downloaded_years:
                total_skipped += 1
                continue
            
            success = download_year(pair, year)
            
            if success:
                total_downloaded += 1
            else:
                total_failed += 1
            
            # Throttle between years
            log(f"Sleeping {DELAY_BETWEEN_YEARS}s before next year...")
            time.sleep(DELAY_BETWEEN_YEARS)
        
        # Throttle between pairs
        log(f"Sleeping {DELAY_BETWEEN_PAIRS}s before next pair...")
        time.sleep(DELAY_BETWEEN_PAIRS)
    
    log("\n" + "=" * 60)
    log("DOWNLOAD QUEUE COMPLETE")
    log(f"Downloaded: {total_downloaded}")
    log(f"Skipped: {total_skipped}")
    log(f"Failed: {total_failed}")
    log("=" * 60)


if __name__ == "__main__":
    # Ensure log directory exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Set process priority to low (nice)
    try:
        os.nice(10)  # Lower priority
        log("Process priority lowered (nice=10)")
    except:
        pass
    
    run_download_queue()
