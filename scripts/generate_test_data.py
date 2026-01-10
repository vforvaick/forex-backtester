"""
Generate synthetic forex tick data for testing the backtester.
"""
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_forex_ticks(
    symbol: str = "EURUSD",
    start_date: str = "2024-01-01",
    days: int = 30,
    ticks_per_day: int = 10000,
    base_price: float = 1.1000,
    volatility: float = 0.0005,
    spread_pips: float = 0.5,
    output_dir: Path = None
) -> pl.DataFrame:
    """
    Generate synthetic tick data with realistic forex characteristics.
    
    Args:
        symbol: Currency pair (e.g., EURUSD)
        start_date: Start date in YYYY-MM-DD format
        days: Number of days to generate
        ticks_per_day: Average ticks per day
        base_price: Starting mid price
        volatility: Daily price volatility
        spread_pips: Average spread in pips
        output_dir: Output directory for parquet file
    
    Returns:
        DataFrame with tick data
    """
    np.random.seed(42)  # For reproducibility
    
    start_dt = datetime.fromisoformat(start_date)
    total_ticks = days * ticks_per_day
    
    # Generate timestamps with realistic distribution (more ticks during active hours)
    timestamps = []
    current_time = start_dt
    
    for day in range(days):
        day_start = start_dt + timedelta(days=day)
        # Generate random intervals in seconds (average 8.64 seconds for 10k ticks/day)
        intervals = np.random.exponential(86400 / ticks_per_day, ticks_per_day)
        
        day_time = 0
        for interval in intervals:
            day_time += interval
            if day_time >= 86400:
                break
            timestamps.append(day_start + timedelta(seconds=day_time))
    
    n_ticks = len(timestamps)
    
    # Generate price movements (random walk with drift)
    returns = np.random.normal(0, volatility / np.sqrt(ticks_per_day), n_ticks)
    
    # Add some trending behavior
    trend = np.sin(np.linspace(0, 4 * np.pi, n_ticks)) * 0.002
    returns = returns + trend / n_ticks
    
    # Calculate mid prices
    mid_prices = base_price * np.exp(np.cumsum(returns))
    
    # Add spread
    pip_size = 0.0001
    half_spread = spread_pips * pip_size / 2
    
    # Variable spread (wider during low activity)
    spread_variation = np.random.uniform(0.8, 1.2, n_ticks)
    spreads = half_spread * spread_variation
    
    ask_prices = mid_prices + spreads
    bid_prices = mid_prices - spreads
    
    # Generate volumes for bid and ask separately
    bid_volumes = np.random.exponential(500000, n_ticks).astype(int)
    ask_volumes = np.random.exponential(500000, n_ticks).astype(int)
    
    # Create DataFrame with all required columns
    df = pl.DataFrame({
        "timestamp": timestamps,
        "bid": bid_prices,
        "ask": ask_prices,
        "bid_vol": bid_volumes,
        "ask_vol": ask_volumes,
        "volume": bid_volumes + ask_volumes,
        "mid": mid_prices
    })
    
    # Save to parquet
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{symbol}_2024.parquet"
        df.write_parquet(output_path)
        print(f"✓ Generated {n_ticks:,} ticks for {symbol}")
        print(f"  Date range: {timestamps[0]} to {timestamps[-1]}")
        print(f"  Price range: {bid_prices.min():.5f} - {ask_prices.max():.5f}")
        print(f"  Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate test data
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "parquet" / "EURUSD"
    
    df = generate_forex_ticks(
        symbol="EURUSD",
        start_date="2024-01-01",
        days=30,
        ticks_per_day=10000,
        output_dir=output_dir
    )
    
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"\nSample data:")
    print(df.head())
