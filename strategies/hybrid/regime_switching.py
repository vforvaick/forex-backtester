"""
Regime Switching Strategy

Dynamically switches between trend-following and mean-reversion strategies
based on detected market regime (trending vs ranging, high vs low volatility).
"""

from typing import Dict
from dataclasses import dataclass
from enum import Enum

import polars as pl


class MarketRegime(Enum):
    TRENDING_HIGH_VOL = "trending_high_vol"
    TRENDING_LOW_VOL = "trending_low_vol"
    RANGING_HIGH_VOL = "ranging_high_vol"
    RANGING_LOW_VOL = "ranging_low_vol"


@dataclass
class Strategy:
    """Market regime detection and adaptive strategy switching."""
    
    # ADX/trend parameters
    adx_period: int = 14
    adx_threshold: float = 25.0  # Above = trending, below = ranging
    
    # Volatility parameters
    atr_period: int = 14
    vol_lookback: int = 100  # Lookback for percentile calculation
    vol_threshold: float = 50.0  # Percentile threshold (50 = median)
    
    # Strategy parameters
    trend_period: int = 20  # Momentum lookback for trend strategy
    mean_period: int = 20  # Mean lookback for reversion strategy
    mean_deviation: float = 0.01  # 1% deviation threshold
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest."""
        # Calculate mid price if not present
        if "mid" not in data.columns:
            data = data.with_columns([
                ((pl.col("ask") + pl.col("bid")) / 2).alias("mid")
            ])
        
        data = self._detect_regime(data)
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _calculate_true_range(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate True Range for ATR."""
        # For tick data, approximate TR as high-low of rolling window
        return data.with_columns([
            pl.col("mid").rolling_max(window_size=5).alias("high_5"),
            pl.col("mid").rolling_min(window_size=5).alias("low_5"),
        ]).with_columns([
            (pl.col("high_5") - pl.col("low_5")).alias("true_range"),
        ])
    
    def _calculate_directional_movement(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate Directional Movement Index components."""
        # +DM: current high - previous high (if positive and > -DM)
        # -DM: previous low - current low (if positive and > +DM)
        
        return data.with_columns([
            (pl.col("high_5") - pl.col("high_5").shift(1)).alias("up_move"),
            (pl.col("low_5").shift(1) - pl.col("low_5")).alias("down_move"),
        ]).with_columns([
            pl.when((pl.col("up_move") > pl.col("down_move")) & (pl.col("up_move") > 0))
            .then(pl.col("up_move"))
            .otherwise(0)
            .alias("plus_dm"),
            
            pl.when((pl.col("down_move") > pl.col("up_move")) & (pl.col("down_move") > 0))
            .then(pl.col("down_move"))
            .otherwise(0)
            .alias("minus_dm"),
        ])
    
    def _detect_regime(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect market regime: trending vs ranging, high vs low volatility."""
        period = self.adx_period
        
        # Calculate True Range and ATR
        data = self._calculate_true_range(data)
        data = data.with_columns([
            pl.col("true_range").rolling_mean(window_size=period).alias("atr"),
        ])
        
        # Calculate Directional Movement
        data = self._calculate_directional_movement(data)
        
        # Smoothed DI+ and DI-
        data = data.with_columns([
            pl.col("plus_dm").rolling_mean(window_size=period).alias("smooth_plus_dm"),
            pl.col("minus_dm").rolling_mean(window_size=period).alias("smooth_minus_dm"),
        ])
        
        # Calculate DI+ and DI-
        data = data.with_columns([
            (100 * pl.col("smooth_plus_dm") / (pl.col("atr") + 0.00001)).alias("di_plus"),
            (100 * pl.col("smooth_minus_dm") / (pl.col("atr") + 0.00001)).alias("di_minus"),
        ])
        
        # Calculate DX and ADX
        data = data.with_columns([
            (100 * (pl.col("di_plus") - pl.col("di_minus")).abs() / 
             (pl.col("di_plus") + pl.col("di_minus") + 0.00001)).alias("dx"),
        ]).with_columns([
            pl.col("dx").rolling_mean(window_size=period).alias("adx"),
        ])
        
        # Calculate volatility percentile
        data = data.with_columns([
            pl.col("atr").rolling_quantile(
                quantile=0.5, 
                window_size=self.vol_lookback
            ).alias("atr_median"),
        ]).with_columns([
            # High volatility = ATR above median
            (pl.col("atr") > pl.col("atr_median")).alias("high_volatility"),
        ])
        
        # Determine regime
        data = data.with_columns([
            (pl.col("adx") > self.adx_threshold).alias("trending"),
        ])
        
        return data
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate regime-adaptive signals."""
        # Detect regime if not present
        if "trending" not in data.columns:
            data = self._detect_regime(data)
        
        # Trend-following signal: momentum-based
        trend_signal = (
            pl.when(pl.col("mid") > pl.col("mid").shift(self.trend_period))
            .then(1)
            .when(pl.col("mid") < pl.col("mid").shift(self.trend_period))
            .then(-1)
            .otherwise(0)
        )
        
        # Mean-reversion signal: deviation from moving average
        data = data.with_columns([
            pl.col("mid").rolling_mean(window_size=self.mean_period).alias("ma"),
        ])
        
        range_signal = (
            pl.when(pl.col("mid") < pl.col("ma") * (1 - self.mean_deviation))
            .then(1)  # Buy when below mean
            .when(pl.col("mid") > pl.col("ma") * (1 + self.mean_deviation))
            .then(-1)  # Sell when above mean
            .otherwise(0)
        )
        
        # Switch based on regime
        # Trending + High Vol: aggressive trend following
        # Trending + Low Vol: moderate trend following  
        # Ranging + High Vol: avoid (choppy)
        # Ranging + Low Vol: mean reversion
        
        signal_expr = (
            pl.when(pl.col("trending") & pl.col("high_volatility"))
            .then(trend_signal)  # Strong trend moves
            .when(pl.col("trending") & ~pl.col("high_volatility"))
            .then(trend_signal)  # Steady trends
            .when(~pl.col("trending") & ~pl.col("high_volatility"))
            .then(range_signal)  # Low vol ranging = mean revert
            .otherwise(0)  # High vol ranging = stay out
        ).alias("signal")
        
        return data.select(signal_expr).to_series()
    
    def _calculate_metrics(self, data: pl.DataFrame, signals: pl.Series) -> Dict[str, float]:
        """Calculate metrics with transaction costs."""
        from strategies.base import calculate_metrics_with_costs
        return calculate_metrics_with_costs(data, signals, symbol="EURUSD")

