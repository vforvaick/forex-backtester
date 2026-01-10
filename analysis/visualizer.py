"""
Strategy Visualizer Module

Generates equity curves, drawdown charts, and trade distribution heatmaps.
Uses Plotly for interactive HTML output and Matplotlib for static PNGs.
"""

import polars as pl
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class VisualizationConfig:
    """Configuration for visualization output."""
    output_dir: str = "results/charts"
    format: str = "html"  # html, png, or both
    theme: str = "plotly_dark"


class StrategyVisualizer:
    """
    Generates visual analysis of backtest results.
    
    Charts:
        - Equity curve (cumulative returns)
        - Drawdown chart (peak-to-trough)
        - Trade distribution heatmap (by hour/day)
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize visualizer with optional config."""
        self.config = config or VisualizationConfig()
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_trades_from_journal(self, db_path: str, run_id: int) -> pl.DataFrame:
        """Load trade data from journal database."""
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get run info
        cursor.execute("""
            SELECT strategy_name, pair, start_date, end_date, metrics
            FROM backtest_runs WHERE id = ?
        """, (run_id,))
        
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Run ID {run_id} not found")
        
        strategy_name, pair, start_date, end_date, metrics_json = row
        metrics = json.loads(metrics_json) if metrics_json else {}
        
        # For now, we'll generate synthetic equity curve from metrics
        # since we don't store individual trades in the journal
        conn.close()
        
        return {
            "strategy_name": strategy_name,
            "pair": pair,
            "start_date": start_date,
            "end_date": end_date,
            "metrics": metrics
        }
    
    def generate_equity_curve(
        self, 
        returns: pl.Series, 
        timestamps: pl.Series,
        strategy_name: str,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate equity curve chart.
        
        Args:
            returns: Series of period returns
            timestamps: Series of timestamps
            strategy_name: Name for chart title
            save_path: Optional custom save path
        
        Returns:
            Path to saved chart
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not installed. Run: pip install plotly")
            return self._generate_matplotlib_equity(returns, timestamps, strategy_name, save_path)
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cum_prod()
        
        # Create figure with secondary y-axis for drawdown
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{strategy_name} - Equity Curve', 'Drawdown')
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=timestamps.to_list(),
                y=cumulative.to_list(),
                mode='lines',
                name='Equity',
                line=dict(color='#00ff88', width=2)
            ),
            row=1, col=1
        )
        
        # Calculate and plot drawdown
        rolling_max = cumulative.cum_max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        fig.add_trace(
            go.Scatter(
                x=timestamps.to_list(),
                y=drawdown.to_list(),
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='#ff4444', width=1)
            ),
            row=2, col=1
        )
        
        # Style
        fig.update_layout(
            template=self.config.theme,
            title=dict(text=f'{strategy_name} Performance Analysis', x=0.5),
            showlegend=True,
            height=700
        )
        
        fig.update_yaxes(title_text="Equity Multiplier", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", tickformat='.1%', row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        # Save
        if save_path is None:
            save_path = Path(self.config.output_dir) / f"{strategy_name.replace(' ', '_')}_equity.html"
        
        fig.write_html(str(save_path))
        print(f"✓ Saved equity chart to: {save_path}")
        
        return str(save_path)
    
    def _generate_matplotlib_equity(
        self,
        returns: pl.Series,
        timestamps: pl.Series,
        strategy_name: str,
        save_path: Optional[str] = None
    ) -> str:
        """Fallback to matplotlib if plotly not available."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Neither Plotly nor Matplotlib installed.")
            return ""
        
        cumulative = (1 + returns).cum_prod().to_list()
        dates = timestamps.to_list()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Equity curve
        ax1.plot(dates, cumulative, color='#00ff88', linewidth=1.5)
        ax1.fill_between(dates, 1, cumulative, alpha=0.3, color='#00ff88')
        ax1.set_ylabel('Equity Multiplier')
        ax1.set_title(f'{strategy_name} - Equity Curve')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        cum_series = pl.Series(cumulative)
        rolling_max = cum_series.cum_max().to_list()
        drawdown = [(c - m) / m if m > 0 else 0 for c, m in zip(cumulative, rolling_max)]
        
        ax2.fill_between(dates, 0, drawdown, color='#ff4444', alpha=0.7)
        ax2.set_ylabel('Drawdown %')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = Path(self.config.output_dir) / f"{strategy_name.replace(' ', '_')}_equity.png"
        
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved equity chart to: {save_path}")
        return str(save_path)
    
    def generate_trade_heatmap(
        self,
        trade_hours: list[int],
        trade_days: list[int],
        trade_pnl: list[float],
        strategy_name: str,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate trade distribution heatmap.
        
        Args:
            trade_hours: List of hours (0-23) when trades occurred
            trade_days: List of days (0-6, Mon-Sun) when trades occurred
            trade_pnl: List of PnL for each trade
            strategy_name: Name for chart title
            save_path: Optional custom save path
        
        Returns:
            Path to saved chart
        """
        try:
            import plotly.express as px
            import numpy as np
        except ImportError:
            print("Plotly not installed. Skipping heatmap.")
            return ""
        
        # Create hour x day matrix
        matrix = [[0.0 for _ in range(24)] for _ in range(7)]
        counts = [[0 for _ in range(24)] for _ in range(7)]
        
        for hour, day, pnl in zip(trade_hours, trade_days, trade_pnl):
            if 0 <= hour < 24 and 0 <= day < 7:
                matrix[day][hour] += pnl
                counts[day][hour] += 1
        
        # Average PnL per cell
        for d in range(7):
            for h in range(24):
                if counts[d][h] > 0:
                    matrix[d][h] /= counts[d][h]
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hours = [f'{h:02d}:00' for h in range(24)]
        
        fig = px.imshow(
            matrix,
            labels=dict(x="Hour", y="Day", color="Avg PnL"),
            x=hours,
            y=days,
            color_continuous_scale='RdYlGn',
            title=f'{strategy_name} - Trade Distribution Heatmap'
        )
        
        fig.update_layout(template=self.config.theme)
        
        if save_path is None:
            save_path = Path(self.config.output_dir) / f"{strategy_name.replace(' ', '_')}_heatmap.html"
        
        fig.write_html(str(save_path))
        print(f"✓ Saved heatmap to: {save_path}")
        
        return str(save_path)
    
    def generate_metrics_summary(
        self,
        metrics: dict,
        strategy_name: str,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate metrics summary card.
        
        Args:
            metrics: Dict with sharpe, sortino, etc.
            strategy_name: Name for chart title
            save_path: Optional custom save path
        
        Returns:
            Path to saved chart
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not installed. Skipping metrics card.")
            return ""
        
        # Create gauge charts for key metrics
        fig = go.Figure()
        
        # Sharpe gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics.get("sharpe", 0),
            domain={'x': [0, 0.3], 'y': [0.5, 1]},
            title={'text': "Sharpe"},
            gauge={
                'axis': {'range': [-2, 3]},
                'bar': {'color': "#00ff88"},
                'steps': [
                    {'range': [-2, 0], 'color': "#ff4444"},
                    {'range': [0, 1], 'color': "#ffaa00"},
                    {'range': [1, 3], 'color': "#00ff88"}
                ]
            }
        ))
        
        # Max Drawdown gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=abs(metrics.get("max_drawdown", 0)) * 100,
            domain={'x': [0.35, 0.65], 'y': [0.5, 1]},
            title={'text': "Max DD %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ff4444"},
                'steps': [
                    {'range': [0, 20], 'color': "#00ff88"},
                    {'range': [20, 40], 'color': "#ffaa00"},
                    {'range': [40, 100], 'color': "#ff4444"}
                ]
            }
        ))
        
        # Profit Factor gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics.get("profit_factor", 0),
            domain={'x': [0.7, 1], 'y': [0.5, 1]},
            title={'text': "Profit Factor"},
            gauge={
                'axis': {'range': [0, 3]},
                'bar': {'color': "#00ff88"},
                'steps': [
                    {'range': [0, 1], 'color': "#ff4444"},
                    {'range': [1, 1.5], 'color': "#ffaa00"},
                    {'range': [1.5, 3], 'color': "#00ff88"}
                ]
            }
        ))
        
        # Stats table
        stats = [
            f"Win Rate: {metrics.get('win_rate', 0):.1%}",
            f"Total Trades: {metrics.get('total_trades', 0):,}",
            f"Total Return: {metrics.get('total_return', 0):.2%}",
            f"Sortino: {metrics.get('sortino', 0):.2f}",
            f"Calmar: {metrics.get('calmar', 0):.2f}"
        ]
        
        fig.add_annotation(
            x=0.5, y=0.25,
            text="<br>".join(stats),
            showarrow=False,
            font=dict(size=14, color='white'),
            align='center'
        )
        
        fig.update_layout(
            template=self.config.theme,
            title=dict(text=f'{strategy_name} - Metrics Summary', x=0.5),
            height=500
        )
        
        if save_path is None:
            save_path = Path(self.config.output_dir) / f"{strategy_name.replace(' ', '_')}_metrics.html"
        
        fig.write_html(str(save_path))
        print(f"✓ Saved metrics card to: {save_path}")
        
        return str(save_path)


def visualize_backtest(
    returns: pl.Series,
    timestamps: pl.Series,
    metrics: dict,
    strategy_name: str,
    output_dir: str = "results/charts"
) -> list[str]:
    """
    Convenience function to generate all visualizations.
    
    Args:
        returns: Series of period returns
        timestamps: Series of timestamps
        metrics: Dict with performance metrics
        strategy_name: Name for chart titles
        output_dir: Directory for output files
    
    Returns:
        List of paths to generated charts
    """
    config = VisualizationConfig(output_dir=output_dir)
    viz = StrategyVisualizer(config)
    
    paths = []
    
    # Equity curve
    path = viz.generate_equity_curve(returns, timestamps, strategy_name)
    if path:
        paths.append(path)
    
    # Metrics summary
    path = viz.generate_metrics_summary(metrics, strategy_name)
    if path:
        paths.append(path)
    
    return paths


if __name__ == "__main__":
    import numpy as np
    from datetime import datetime, timedelta
    
    # Demo with synthetic data
    np.random.seed(42)
    n_periods = 252  # 1 year of daily data
    
    # Generate random returns
    returns = np.random.normal(0.001, 0.02, n_periods)
    returns = pl.Series("returns", returns)
    
    # Generate timestamps
    start = datetime(2024, 1, 1)
    timestamps = pl.Series("timestamp", [start + timedelta(days=i) for i in range(n_periods)])
    
    # Sample metrics
    metrics = {
        "sharpe": 1.2,
        "sortino": 1.8,
        "max_drawdown": 0.15,
        "profit_factor": 1.45,
        "win_rate": 0.52,
        "total_trades": 156,
        "total_return": 0.24,
        "calmar": 1.6
    }
    
    # Generate all charts
    paths = visualize_backtest(
        returns=returns,
        timestamps=timestamps,
        metrics=metrics,
        strategy_name="MA Crossover Demo"
    )
    
    print(f"\nGenerated {len(paths)} charts:")
    for p in paths:
        print(f"  - {p}")
