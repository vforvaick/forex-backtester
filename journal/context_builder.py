"""
LLM Context Builder

Builds context from journal history for intelligent LLM evaluations.
Feeds related runs and discovered insights to prevent redundant experiments.
"""

from typing import Dict, List, Optional
import json

from .database import get_related_runs, get_insights


def build_llm_context(
    current_strategy: str,
    current_metrics: Dict,
    history_limit: int = 20
) -> str:
    """
    Build context string for LLM evaluation.
    
    Includes:
    - Related historical runs
    - Discovered patterns/insights
    - Current backtest results
    
    Args:
        current_strategy: Name of strategy being evaluated
        current_metrics: Metrics from current backtest
        history_limit: Max historical runs to include
    
    Returns:
        Formatted context string for LLM prompt
    """
    # Get historical context
    related_runs = get_related_runs(current_strategy, limit=history_limit)
    insights = get_insights(min_confidence=0.6, limit=10)
    
    # Format historical runs
    history_section = format_run_history(related_runs)
    
    # Format insights
    insights_section = format_insights(insights)
    
    # Format current results
    current_section = format_current_metrics(current_strategy, current_metrics)
    
    context = f"""
## Historical Context

### Previous Runs for Similar Strategies ({len(related_runs)} runs)
{history_section}

### Discovered Patterns (High Confidence)
{insights_section}

---

## Current Backtest Results
{current_section}

---

## Evaluation Request

Based on the historical context and current results:
1. Is this strategy showing genuine edge or overfitting?
2. What specific parameter adjustments would you recommend?
3. Are there any concerning patterns we should investigate?
4. Should we mark this as "promising", "reject", or "needs_refinement"?

Provide your analysis in structured JSON format:
```json
{{
  "verdict": "promising|reject|needs_refinement",
  "confidence": 0.0-1.0,
  "analysis": "Your detailed analysis...",
  "concerns": ["list of concerns"],
  "recommendations": [
    {{"param": "name", "current": value, "suggested": value, "reasoning": "..."}}
  ],
  "next_experiments": ["suggested follow-up tests"]
}}
```
"""
    
    return context


def format_run_history(runs: List[Dict]) -> str:
    """Format historical runs for LLM context."""
    if not runs:
        return "_No historical runs found for this strategy._"
    
    lines = []
    for run in runs[:10]:  # Limit to 10 most recent
        params = json.loads(run.get("tuning_params") or "{}")
        params_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:3])
        
        lines.append(
            f"- **{run['strategy_name']}** ({run['run_timestamp'][:10]}): "
            f"Sharpe={run['sharpe']:.2f}, DD={run['max_drawdown']:.1%}, "
            f"WR={run['win_rate']:.1%} → {run['verdict'] or 'pending'}\n"
            f"  Params: {params_str}"
        )
    
    return "\n".join(lines)


def format_insights(insights: List[Dict]) -> str:
    """Format insights for LLM context."""
    if not insights:
        return "_No high-confidence insights discovered yet._"
    
    lines = []
    for insight in insights:
        lines.append(
            f"- **{insight['pattern']}** (confidence: {insight['confidence']:.0%})"
        )
    
    return "\n".join(lines)


def format_current_metrics(strategy: str, metrics: Dict) -> str:
    """Format current backtest metrics."""
    return f"""
**Strategy**: {strategy}

| Metric | Value |
|--------|-------|
| Sharpe Ratio | {metrics.get('sharpe', 'N/A'):.2f} |
| Sortino Ratio | {metrics.get('sortino', 'N/A'):.2f} |
| Max Drawdown | {metrics.get('max_drawdown', 0):.1%} |
| Win Rate | {metrics.get('win_rate', 0):.1%} |
| Profit Factor | {metrics.get('profit_factor', 'N/A'):.2f} |
| Total Trades | {metrics.get('total_trades', 0)} |
| Total Return | {metrics.get('total_return', 0):.1%} |

**Tuning Parameters**: {json.dumps(metrics.get('params', {}), indent=2)}
"""


def extract_insights_from_evaluation(
    evaluation_json: Dict,
    run_id: int
) -> List[Dict]:
    """
    Extract potential insights from LLM evaluation.
    
    Returns list of insights to potentially add to knowledge base.
    """
    insights = []
    
    # Extract from concerns
    for concern in evaluation_json.get("concerns", []):
        if len(concern) > 20:  # Substantial insight
            insights.append({
                "pattern": concern,
                "evidence_runs": [run_id],
                "confidence": 0.5  # Initial confidence
            })
    
    # Extract from recommendations with strong reasoning
    for rec in evaluation_json.get("recommendations", []):
        if rec.get("reasoning"):
            pattern = f"{rec['param']}: {rec['reasoning']}"
            insights.append({
                "pattern": pattern,
                "evidence_runs": [run_id],
                "confidence": 0.4
            })
    
    return insights
