"""
CLIProxy LLM Evaluator

Integrates with your CLIProxy for LLM-based strategy evaluation.
Supports Claude, GPT, and Gemini models via your proxy.
"""

import json
import httpx
from typing import Dict, Optional, Any
from pathlib import Path
import sys

# Add parent to path for journal import
sys.path.insert(0, str(Path(__file__).parent.parent))

from journal import build_llm_context, save_llm_evaluation, extract_insights_from_evaluation, add_insight


# CLIProxy configuration (internal VCN endpoint from binobot)
CLIPROXY_URL = "http://172.27.1.12:8317/v1/chat/completions"
CLIPROXY_API_KEY = "cliproxy-secret-2024"
DEFAULT_MODEL = "gemini-2.5-flash-lite"  # Fast and available model


SYSTEM_PROMPT = """You are an expert Quantitative Trading Strategist and Risk Manager.

Your role is to evaluate forex backtesting results and provide actionable recommendations.
You have access to historical context from previous experiments to avoid redundant suggestions.

IMPORTANT GUIDELINES:
1. Focus on hypothesis-driven improvements based on market logic, not arbitrary curve fitting
2. Consider market regimes (trending vs ranging, high vs low volatility)
3. Flag potential overfitting if metrics look too good
4. Suggest concrete parameter changes with clear reasoning
5. Learn from historical failures - don't repeat experiments that already failed

Always respond in valid JSON format as specified."""


async def evaluate_strategy(
    strategy_name: str,
    metrics: Dict[str, Any],
    run_id: int,
    model: str = DEFAULT_MODEL,
    cliproxy_url: str = CLIPROXY_URL
) -> Dict:
    """
    Evaluate backtest results using LLM via CLIProxy.
    
    Args:
        strategy_name: Name of the strategy
        metrics: Backtest metrics dictionary
        run_id: Database run ID for reference
        model: LLM model to use
        cliproxy_url: CLIProxy endpoint URL
    
    Returns:
        Parsed evaluation response
    """
    # Build context with historical data
    context = build_llm_context(strategy_name, metrics)
    
    # Prepare request
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ],
        "temperature": 0.3,  # Lower for more deterministic analysis
        "max_tokens": 2000
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            headers = {"Authorization": f"Bearer {CLIPROXY_API_KEY}"}
            response = await client.post(cliproxy_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse JSON from response
            evaluation = parse_llm_response(content)
            
            # Save to journal
            eval_id = save_llm_evaluation(
                run_id=run_id,
                model=model,
                analysis=evaluation.get("analysis", ""),
                suggested_params={r["param"]: r["suggested"] for r in evaluation.get("recommendations", [])},
                reasoning=json.dumps(evaluation.get("recommendations", []))
            )
            
            # Extract and save insights
            new_insights = extract_insights_from_evaluation(evaluation, run_id)
            for insight in new_insights:
                add_insight(
                    pattern=insight["pattern"],
                    evidence_runs=insight["evidence_runs"],
                    confidence=insight["confidence"]
                )
            
            evaluation["eval_id"] = eval_id
            return evaluation
            
        except httpx.HTTPError as e:
            return {
                "error": str(e),
                "verdict": "error",
                "analysis": f"Failed to evaluate: {e}"
            }


def parse_llm_response(content: str) -> Dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    # Try to extract JSON from code blocks
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        json_str = content[start:end].strip()
    elif "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        json_str = content[start:end].strip()
    else:
        json_str = content.strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {
            "verdict": "error",
            "analysis": content,
            "concerns": ["Failed to parse structured response"],
            "recommendations": []
        }


def evaluate_strategy_sync(
    strategy_name: str,
    metrics: Dict[str, Any],
    run_id: int,
    model: str = DEFAULT_MODEL
) -> Dict:
    """Synchronous wrapper for evaluate_strategy."""
    import asyncio
    return asyncio.run(evaluate_strategy(strategy_name, metrics, run_id, model))


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate strategy with LLM")
    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument("--metrics", required=True, help="JSON file with metrics")
    parser.add_argument("--run-id", type=int, required=True, help="Database run ID")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model")
    
    args = parser.parse_args()
    
    with open(args.metrics) as f:
        metrics = json.load(f)
    
    result = evaluate_strategy_sync(args.strategy, metrics, args.run_id, args.model)
    print(json.dumps(result, indent=2))
