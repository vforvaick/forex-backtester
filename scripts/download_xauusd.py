#!/usr/bin/env python3
"""
Download all XAUUSD tick data from Dukascopy.
Run with: nohup python3 scripts/download_xauusd.py > logs/xauusd_download.log 2>&1 &
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from duka.app import app as duka_download
from duka.core.utils import TimeFrame
import polars as pl

OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'parquet' / 'XAUUSD'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# XAUUSD available from ~2010 on Dukascopy
START_YEAR = 2010
END_YEAR = 2024


def download_year(pair: str, year: int):
    """Download one year of data."""
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    
    # Don't download future dates
    if end > date.today():
        end = date.today() - timedelta(days=1)
    
    if start > date.today():
        print(f"Skipping {year} - future date")
        return None
    
    parquet_path = OUTPUT_DIR / f'{year}.parquet'
    if parquet_path.exists():
        print(f'{year} already exists, skipping...')
        return parquet_path
    
    print(f'\n{"="*50}')
    print(f'Downloading {pair} {year}...')
    print(f'{"="*50}')
    sys.stdout.flush()
    
    temp_dir = Path('/tmp/duka_download')
    temp_dir.mkdir(exist_ok=True)
    
    try:
        duka_download(
            [pair],
            start,
            end,
            4,  # threads
            TimeFrame.TICK,
            str(temp_dir),
            True  # header
        )
        
        # Find and convert CSV
        csvs = list(temp_dir.glob(f'{pair}*.csv'))
        if csvs:
            csv_path = sorted(csvs, key=os.path.getmtime)[-1]
            print(f'Converting {csv_path.name} to Parquet...')
            
            df = pl.read_csv(csv_path, try_parse_dates=True)
            
            # Standardize column names
            if 'time' in df.columns:
                df = df.rename({'time': 'timestamp'})
            
            # Add mid price and spread
            df = df.with_columns([
                ((pl.col('ask') + pl.col('bid')) / 2).alias('mid'),
                (pl.col('ask') - pl.col('bid')).alias('spread')
            ])
            
            df.write_parquet(parquet_path)
            print(f'Saved: {parquet_path} ({len(df):,} rows)')
            sys.stdout.flush()
            
            # Cleanup
            csv_path.unlink()
            return parquet_path
        else:
            print(f'No CSV found for {year}')
            return None
            
    except Exception as e:
        print(f'Error downloading {year}: {e}')
        return None


def main():
    print('='*60)
    print('XAUUSD Full Historical Download')
    print(f'Years: {START_YEAR} - {END_YEAR}')
    print(f'Output: {OUTPUT_DIR}')
    print('='*60)
    sys.stdout.flush()
    
    successful = []
    failed = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        result = download_year('XAUUSD', year)
        if result:
            successful.append(year)
        else:
            failed.append(year)
    
    print('\n' + '='*60)
    print('DOWNLOAD COMPLETE')
    print('='*60)
    print(f'Successful: {len(successful)} years - {successful}')
    print(f'Failed: {len(failed)} years - {failed}')
    
    # Show total size
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob('*.parquet'))
    print(f'Total data size: {total_size / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    main()
