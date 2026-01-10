"""
Progress Reporting Module

Provides a heartbeat system for granular progress reporting during backtests.
Uses Rich library for live dashboard updates.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime


@dataclass
class BacktestProgress:
    """Tracks progress of a single backtest run."""
    strategy_name: str
    total_years: int = 0
    current_year: int = 0
    current_stage: str = "initializing"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    @property
    def progress_pct(self) -> float:
        if self.total_years == 0:
            return 0.0
        return (self.current_year / self.total_years) * 100
    
    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "progress": self.progress_pct,
            "stage": self.current_stage,
            "elapsed": self.elapsed_seconds,
            "year": self.current_year,
            "total_years": self.total_years
        }


class Heartbeat:
    """
    Heartbeat system for reporting backtest progress.
    
    Writes status to JSON files that can be monitored by the dashboard.
    """
    
    def __init__(self, output_dir: Path = Path("results/live")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress: Dict[str, BacktestProgress] = {}
    
    def start(self, strategy_name: str, total_years: int = 1) -> None:
        """Register a new backtest run."""
        self.progress[strategy_name] = BacktestProgress(
            strategy_name=strategy_name,
            total_years=total_years
        )
        self._write_status(strategy_name)
    
    def update(self, strategy_name: str, year: int, stage: str) -> None:
        """Update progress for a backtest run."""
        if strategy_name in self.progress:
            self.progress[strategy_name].current_year = year
            self.progress[strategy_name].current_stage = stage
            self._write_status(strategy_name)
    
    def complete(self, strategy_name: str, status: str = "done") -> None:
        """Mark a backtest run as complete."""
        if strategy_name in self.progress:
            self.progress[strategy_name].current_stage = status
            self.progress[strategy_name].end_time = time.time()
            self._write_status(strategy_name)
    
    def _write_status(self, strategy_name: str) -> None:
        """Write status to file."""
        if strategy_name in self.progress:
            status_file = self.output_dir / f"{strategy_name}.json"
            with open(status_file, "w") as f:
                json.dump(self.progress[strategy_name].to_dict(), f)
    
    def read_all(self) -> Dict[str, dict]:
        """Read all status files."""
        statuses = {}
        for f in self.output_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    statuses[f.stem] = json.load(fp)
            except (json.JSONDecodeError, IOError):
                pass
        return statuses
    
    def clear(self) -> None:
        """Clear all status files."""
        for f in self.output_dir.glob("*.json"):
            f.unlink()


def create_live_dashboard():
    """
    Create a Rich Live dashboard for monitoring sweep progress.
    
    Returns a context manager that updates the display.
    """
    try:
        from rich.live import Live
        from rich.table import Table
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
        
        console = Console()
        
        def make_table(heartbeat: Heartbeat, completed: int, total: int) -> Table:
            table = Table(title="🚀 Sweep Progress Dashboard")
            table.add_column("Strategy", style="cyan", no_wrap=True)
            table.add_column("Progress", justify="center")
            table.add_column("Stage", style="yellow")
            table.add_column("Elapsed", style="green")
            
            statuses = heartbeat.read_all()
            for name, status in sorted(statuses.items()):
                pct = status.get("progress", 0)
                bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                elapsed = f"{status.get('elapsed', 0):.1f}s"
                table.add_row(
                    name[:25],
                    f"[{bar}] {pct:.0f}%",
                    status.get("stage", "?"),
                    elapsed
                )
            
            table.add_section()
            table.add_row(
                f"[bold]Total[/bold]",
                f"[bold]{completed}/{total}[/bold]",
                "",
                ""
            )
            return table
        
        return Live, make_table, console
        
    except ImportError:
        print("Rich not installed. Using basic progress.")
        return None, None, None
