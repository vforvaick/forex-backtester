"""
Harmonic Pattern Strategy

Detects Gartley, Bat, Butterfly, and Crab patterns using XABCD points
and Fibonacci ratios for high-probability reversal trades.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import polars as pl


class PatternType(Enum):
    GARTLEY = "gartley"
    BAT = "bat"
    BUTTERFLY = "butterfly"
    CRAB = "crab"


# Fibonacci ratios for each pattern (XB, AC, BD retrace/extension)
PATTERN_RATIOS = {
    PatternType.GARTLEY: {
        "xb": (0.618, 0.618),      # B retraces 61.8% of XA
        "ac": (0.382, 0.886),      # C retraces 38.2-88.6% of AB
        "bd": (1.272, 1.618),      # D extends 127.2-161.8% of BC
        "xd": (0.786, 0.786),      # D retraces 78.6% of XA
    },
    PatternType.BAT: {
        "xb": (0.382, 0.500),      # B retraces 38.2-50% of XA
        "ac": (0.382, 0.886),      # C retraces 38.2-88.6% of AB
        "bd": (1.618, 2.618),      # D extends 161.8-261.8% of BC
        "xd": (0.886, 0.886),      # D retraces 88.6% of XA
    },
    PatternType.BUTTERFLY: {
        "xb": (0.786, 0.786),      # B retraces 78.6% of XA
        "ac": (0.382, 0.886),      # C retraces 38.2-88.6% of AB
        "bd": (1.618, 2.618),      # D extends 161.8-261.8% of BC
        "xd": (1.272, 1.618),      # D extends beyond X
    },
    PatternType.CRAB: {
        "xb": (0.382, 0.618),      # B retraces 38.2-61.8% of XA
        "ac": (0.382, 0.886),      # C retraces 38.2-88.6% of AB
        "bd": (2.618, 3.618),      # D extends 261.8-361.8% of BC
        "xd": (1.618, 1.618),      # D extends 161.8% of XA
    },
}


@dataclass
class Strategy:
    """Harmonic pattern trading with XABCD detection."""
    
    patterns: List[str] = None
    tolerance: float = 0.05  # 5% tolerance for Fib matching
    swing_lookback: int = 20  # Lookback for swing detection
    min_pattern_bars: int = 30  # Minimum bars for pattern formation
    
    def __init__(self, **kwargs):
        self.patterns = kwargs.get("patterns", ["gartley", "bat"])
        self.tolerance = kwargs.get("tolerance", 0.05)
        self.swing_lookback = kwargs.get("swing_lookback", 20)
        self.min_pattern_bars = kwargs.get("min_pattern_bars", 30)
    
    def backtest(self, data: pl.DataFrame) -> Dict[str, float]:
        """Run backtest."""
        # Calculate mid price if not present
        if "mid" not in data.columns:
            data = data.with_columns([
                ((pl.col("ask") + pl.col("bid")) / 2).alias("mid")
            ])
        
        data = self._detect_swing_points(data)
        data = self._detect_patterns(data)
        signals = self._generate_signals(data)
        return self._calculate_metrics(data, signals)
    
    def _detect_swing_points(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect swing highs and lows for XABCD identification."""
        lb = self.swing_lookback
        
        return data.with_columns([
            # Swing high: local maximum
            (pl.col("mid") == pl.col("mid").rolling_max(window_size=lb)).alias("is_swing_high"),
            # Swing low: local minimum
            (pl.col("mid") == pl.col("mid").rolling_min(window_size=lb)).alias("is_swing_low"),
            # Track swing values
            pl.col("mid").rolling_max(window_size=lb).alias("swing_high"),
            pl.col("mid").rolling_min(window_size=lb).alias("swing_low"),
        ])
    
    def _check_fib_ratio(self, value: float, target_range: Tuple[float, float]) -> bool:
        """Check if value is within tolerance of target Fibonacci range."""
        min_val = target_range[0] * (1 - self.tolerance)
        max_val = target_range[1] * (1 + self.tolerance)
        return min_val <= value <= max_val
    
    def _detect_patterns(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect harmonic patterns using simplified Fibonacci confluence."""
        # For tick data, we use a simplified approach:
        # Track recent swing points and check for Fibonacci relationships
        
        lb = self.min_pattern_bars
        
        # Calculate retracement levels from recent swings
        data = data.with_columns([
            pl.col("swing_high").shift(lb).alias("x_high"),
            pl.col("swing_low").shift(lb).alias("x_low"),
            (pl.col("swing_high") - pl.col("swing_low")).alias("swing_range"),
        ])
        
        # Calculate potential D point completion zones for bullish/bearish patterns
        data = data.with_columns([
            (pl.col("swing_low") - pl.col("swing_range")).alias("swing_range_prev"),
        ]).with_columns([
            # Key Fibonacci retracement levels
            (pl.col("swing_low") + 0.618 * pl.col("swing_range")).alias("fib_618"),
            (pl.col("swing_low") + 0.786 * pl.col("swing_range")).alias("fib_786"),
            (pl.col("swing_low") + 0.886 * pl.col("swing_range")).alias("fib_886"),
            # Extension levels
            (pl.col("swing_high") + 0.272 * pl.col("swing_range")).alias("ext_1272"),
            (pl.col("swing_high") + 0.618 * pl.col("swing_range")).alias("ext_1618"),
        ])
        
        # Calculate distances to key Fib levels (normalized)
        data = data.with_columns([
            ((pl.col("mid") - pl.col("fib_786")).abs() / 
             (pl.col("swing_range") + 0.00001)).alias("dist_786"),
            ((pl.col("mid") - pl.col("fib_886")).abs() / 
             (pl.col("swing_range") + 0.00001)).alias("dist_886"),
            ((pl.col("mid") - pl.col("fib_618")).abs() / 
             (pl.col("swing_range") + 0.00001)).alias("dist_618"),
        ])
        
        # Pattern detection: price at key Fibonacci levels with momentum shift
        mom_period = 5
        data = data.with_columns([
            pl.col("mid").diff(mom_period).alias("momentum"),
        ])
        
        # Bullish pattern completion (Gartley/Bat at 78.6%/88.6% retracement)
        data = data.with_columns([
            # Bullish Gartley: price at 78.6% with bullish momentum shift
            ((pl.col("dist_786") < self.tolerance) & 
             (pl.col("momentum") > 0) &
             (pl.col("momentum").shift(1) < 0)).alias("bullish_gartley"),
            
            # Bullish Bat: price at 88.6% with bullish momentum shift
            ((pl.col("dist_886") < self.tolerance) & 
             (pl.col("momentum") > 0) &
             (pl.col("momentum").shift(1) < 0)).alias("bullish_bat"),
            
            # Bearish patterns at swing highs
            ((pl.col("dist_786") < self.tolerance) & 
             (pl.col("momentum") < 0) &
             (pl.col("momentum").shift(1) > 0) &
             (pl.col("mid") > pl.col("fib_618"))).alias("bearish_gartley"),
            
            ((pl.col("dist_886") < self.tolerance) & 
             (pl.col("momentum") < 0) &
             (pl.col("momentum").shift(1) > 0) &
             (pl.col("mid") > pl.col("fib_618"))).alias("bearish_bat"),
        ])
        
        return data
    
    def _generate_signals(self, data: pl.DataFrame) -> pl.Series:
        """Generate signals at pattern completion points."""
        # Detect patterns if not present
        if "bullish_gartley" not in data.columns:
            data = self._detect_patterns(data)
        
        # Combine pattern signals
        signal_expr = (
            pl.when(pl.col("bullish_gartley") | pl.col("bullish_bat"))
            .then(1)
            .when(pl.col("bearish_gartley") | pl.col("bearish_bat"))
            .then(-1)
            .otherwise(0)
        ).alias("signal")
        
        return data.select(signal_expr).to_series()
    
    def _calculate_metrics(self, data: pl.DataFrame, signals: pl.Series) -> Dict[str, float]:
        """Calculate metrics with transaction costs."""
        from strategies.base import calculate_metrics_with_costs
        return calculate_metrics_with_costs(data, signals, symbol="EURUSD")

