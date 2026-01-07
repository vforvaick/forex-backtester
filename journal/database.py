"""
Backtesting Journal Database

SQLite-based storage for backtest results and LLM evaluations.
Provides context for intelligent parameter recommendations.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).parent.parent / "backtest_journal.db"


def get_connection() -> sqlite3.Connection:
    """Get database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database schema."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Core results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            tuning_params TEXT,  -- JSON
            run_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            data_start DATE,
            data_end DATE,
            sharpe REAL,
            sortino REAL,
            max_drawdown REAL,
            win_rate REAL,
            total_trades INTEGER,
            profit_factor REAL,
            calmar REAL,
            total_return REAL,
            verdict TEXT CHECK(verdict IN ('promising', 'reject', 'needs_refinement')),
            notes TEXT
        )
    """)
    
    # LLM feedback history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER REFERENCES backtest_runs(id),
            model_used TEXT,
            analysis TEXT,
            suggested_params TEXT,  -- JSON
            reasoning TEXT,
            followed BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Knowledge base (learned patterns)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern TEXT NOT NULL,
            evidence_runs TEXT,  -- JSON array of run_ids
            confidence REAL DEFAULT 0.5,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_strategy ON backtest_runs(strategy_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_verdict ON backtest_runs(verdict)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_confidence ON insights(confidence)")
    
    conn.commit()
    conn.close()
    print(f"Database initialized: {DB_PATH}")


def save_backtest_run(
    strategy_name: str,
    tuning_params: Dict[str, Any],
    metrics: Dict[str, float],
    data_range: tuple,
    verdict: str = "needs_refinement",
    notes: str = ""
) -> int:
    """
    Save backtest results to journal.
    
    Returns:
        Run ID for reference
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO backtest_runs 
        (strategy_name, tuning_params, data_start, data_end,
         sharpe, sortino, max_drawdown, win_rate, total_trades,
         profit_factor, calmar, total_return, verdict, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        strategy_name,
        json.dumps(tuning_params),
        data_range[0],
        data_range[1],
        metrics.get("sharpe"),
        metrics.get("sortino"),
        metrics.get("max_drawdown"),
        metrics.get("win_rate"),
        metrics.get("total_trades"),
        metrics.get("profit_factor"),
        metrics.get("calmar"),
        metrics.get("total_return"),
        verdict,
        notes
    ))
    
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return run_id


def save_llm_evaluation(
    run_id: int,
    model: str,
    analysis: str,
    suggested_params: Dict[str, Any],
    reasoning: str
) -> int:
    """Save LLM evaluation for a backtest run."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO llm_evaluations 
        (run_id, model_used, analysis, suggested_params, reasoning)
        VALUES (?, ?, ?, ?, ?)
    """, (
        run_id,
        model,
        analysis,
        json.dumps(suggested_params),
        reasoning
    ))
    
    eval_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return eval_id


def get_related_runs(strategy_pattern: str, limit: int = 20) -> List[Dict]:
    """Get recent runs matching strategy pattern."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT r.*, e.reasoning, e.suggested_params
        FROM backtest_runs r
        LEFT JOIN llm_evaluations e ON e.run_id = r.id
        WHERE r.strategy_name LIKE ?
        ORDER BY r.run_timestamp DESC
        LIMIT ?
    """, (f"%{strategy_pattern}%", limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_insights(min_confidence: float = 0.7, limit: int = 10) -> List[Dict]:
    """Get high-confidence insights."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM insights
        WHERE confidence >= ?
        ORDER BY confidence DESC
        LIMIT ?
    """, (min_confidence, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def add_insight(pattern: str, evidence_runs: List[int], confidence: float = 0.5):
    """Add or update an insight in the knowledge base."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if similar pattern exists
    cursor.execute("SELECT id FROM insights WHERE pattern = ?", (pattern,))
    existing = cursor.fetchone()
    
    if existing:
        # Update existing
        cursor.execute("""
            UPDATE insights 
            SET evidence_runs = ?, confidence = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (json.dumps(evidence_runs), confidence, existing["id"]))
    else:
        # Insert new
        cursor.execute("""
            INSERT INTO insights (pattern, evidence_runs, confidence)
            VALUES (?, ?, ?)
        """, (pattern, json.dumps(evidence_runs), confidence))
    
    conn.commit()
    conn.close()


# Initialize on import
if not DB_PATH.exists():
    init_database()
