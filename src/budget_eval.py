"""
Budget-matched evaluation for fair comparison.

Ensures all methods are compared under equivalent compute budgets
(fixed token count), producing Pareto curves of factuality vs cost.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BudgetPoint:
    method: str
    token_budget: int
    factuality: float
    completeness: float
    helpfulness: float
    actual_tokens: float
    latency_s: float


def enforce_token_budget(text: str, budget: int) -> str:
    """Truncate text to at most `budget` tokens (whitespace-split)."""
    tokens = text.split()
    if len(tokens) <= budget:
        return text
    return " ".join(tokens[:budget])


def compute_pareto_front(points: list[BudgetPoint]) -> list[BudgetPoint]:
    """Extract Pareto-optimal points (maximize factuality, minimize tokens)."""
    sorted_pts = sorted(points, key=lambda p: p.actual_tokens)
    pareto = []
    best_fact = -1.0

    for pt in sorted_pts:
        if pt.factuality > best_fact:
            pareto.append(pt)
            best_fact = pt.factuality

    return pareto


def evaluate_under_budget(
    method_results: dict,
    token_budgets: list[int] = None,
) -> dict:
    """
    Re-evaluate all methods under fixed token budgets.

    method_results: {method_name: [{text, factuality, tokens, latency}, ...]}
    Returns Pareto-comparable metrics at each budget level.
    """
    if token_budgets is None:
        token_budgets = [64, 128, 256, 512]

    budget_report = {}

    for budget in token_budgets:
        budget_key = f"budget_{budget}"
        budget_report[budget_key] = {}

        for method, results in method_results.items():
            truncated_facts = []
            actual_tokens_list = []

            for r in results:
                text = r.get("text", "")
                orig_tokens = len(text.split())
                truncated = enforce_token_budget(text, budget)
                actual = min(orig_tokens, budget)

                fact = r.get("factuality", 0.5)
                if orig_tokens > budget:
                    fact *= (actual / max(orig_tokens, 1))

                truncated_facts.append(fact)
                actual_tokens_list.append(actual)

            avg_fact = float(np.mean(truncated_facts)) if truncated_facts else 0.0
            avg_tokens = float(np.mean(actual_tokens_list)) if actual_tokens_list else 0.0
            efficiency = avg_fact / max(avg_tokens / 100.0, 0.01)

            budget_report[budget_key][method] = {
                "factuality": avg_fact,
                "avg_tokens_used": avg_tokens,
                "efficiency": efficiency,
                "n_samples": len(results),
            }

    all_points = []
    for budget in token_budgets:
        budget_key = f"budget_{budget}"
        for method, metrics in budget_report[budget_key].items():
            all_points.append(BudgetPoint(
                method=method,
                token_budget=budget,
                factuality=metrics["factuality"],
                completeness=0.0,
                helpfulness=0.0,
                actual_tokens=metrics["avg_tokens_used"],
                latency_s=0.0,
            ))

    pareto = compute_pareto_front(all_points)
    budget_report["pareto_front"] = [
        {"method": p.method, "budget": p.token_budget,
         "factuality": p.factuality, "tokens": p.actual_tokens}
        for p in pareto
    ]

    return budget_report
