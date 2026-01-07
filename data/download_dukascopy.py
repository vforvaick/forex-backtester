"""
Dukascopy Historical Data Downloader

Downloads tick data from Dukascopy for Forex backtesting.
Uses the 'duka' library for efficient bulk downloads.
"""

import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional

# Note: Install with: pip install duka polars
try:
    from duka.app import app as duka_download
    from duka.core.utils import TimeFrame
except ImportError:
    print("Please install duka: pip install duka")
    duka_download = None
    TimeFrame = None

import polars as pl


# Supported currency pairs
FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"
]

DATA_DIR = Path(__file__).parent / "parquet"


def download_dukascopy(
    pair: str,
    start_date: date,
    end_date: date,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Download tick data from Dukascopy.
    
    Args:
        pair: Currency pair (e.g., "EURUSD")
        start_date: Start date for data download
        end_date: End date for data download
        output_dir: Output directory (default: data/parquet/)
    
    Returns:
        Path to downloaded Parquet file
    """
    if duka_download is None:
        raise ImportError("duka not installed. Run: pip install duka")
    
    output_dir = output_dir or DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download to CSV first (duka's native format)
    csv_path = output_dir / f"{pair}_{start_date.year}.csv"
    
    print(f"Downloading {pair} from {start_date} to {end_date}...")
    duka_download(
        [pair],  # symbols must be a list
        start_date,  # duka expects date objects
        end_date,
        1,  # threads
        TimeFrame.TICK,
        str(output_dir),
        True  # header
    )
    
    # Convert to Parquet with Polars
    # Duka names files like EURUSD-2024_01_01-2024_01_02.csv
    actual_csv = list(output_dir.glob(f"{pair}-*.csv"))
    if not actual_csv:
        # Try fallback naming
        actual_csv = list(output_dir.glob(f"{pair}_*.csv"))
    
    if actual_csv:
        csv_path = sorted(actual_csv, key=os.path.getmtime)[-1]
        parquet_path = convert_to_parquet(csv_path, pair, start_date.year)
        # Cleanup CSV to save space
        csv_path.unlink()
    else:
        print(f"Warning: No CSV found for {pair}")
        return None
    
    return parquet_path


def convert_to_parquet(csv_path: Path, pair: str, year: int) -> Path:
    """Convert Dukascopy CSV to optimized Parquet format."""
    
    parquet_path = DATA_DIR / pair / f"{year}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Lazy loading for memory efficiency
    df = pl.scan_csv(csv_path).with_columns([
        pl.col("time").str.to_datetime().alias("timestamp"),
        pl.col("ask").cast(pl.Float64),
        pl.col("bid").cast(pl.Float64),
        pl.col("ask_volume").cast(pl.Float64).alias("ask_vol"),
        pl.col("bid_volume").cast(pl.Float64).alias("bid_vol"),
    ]).select([
        "timestamp", "bid", "ask", "bid_vol", "ask_vol"
    ])
    
    # Write with compression
    df.collect().write_parquet(
        parquet_path,
        compression="zstd",
        compression_level=3
    )
    
    print(f"Saved: {parquet_path}")
    return parquet_path


def load_data(
    pair: str,
    start_year: int,
    end_year: int
) -> pl.LazyFrame:
    """
    Load tick data from Parquet files (lazy).
    
    Args:
        pair: Currency pair
        start_year: Start year
        end_year: End year (inclusive)
    
    Returns:
        Polars LazyFrame with tick data
    """
    files = []
    for year in range(start_year, end_year + 1):
        path = DATA_DIR / pair / f"{year}.parquet"
        if path.exists():
            files.append(path)
    
    if not files:
        raise FileNotFoundError(f"No data found for {pair}")
    
    # Lazy concatenation
    return pl.concat([pl.scan_parquet(f) for f in files])


if __name__ == "__main__":
    # Example: Download EURUSD for 2020
    download_dukascopy(
        "EURUSD",
        date(2020, 1, 1),
        date(2020, 12, 31)
    )
