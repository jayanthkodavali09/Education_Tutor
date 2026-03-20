"""
cost_tracker.py — Token counting and cost comparison.

Compares:
  - Baseline RAG: sends entire textbook text to LLM (naive approach)
  - Pruned RAG:   sends only top-k retrieved chunks (our approach)
"""

import tiktoken

# GPT-3.5-turbo pricing (as of 2024, USD per 1K tokens)
PRICING = {
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-4":         {"input": 0.03,   "output": 0.06},
    "gpt-4o-mini":   {"input": 0.00015,"output": 0.0006},
}

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-3.5-turbo") -> float:
    p = PRICING.get(model, PRICING["gpt-3.5-turbo"])
    return (input_tokens / 1000) * p["input"] + (output_tokens / 1000) * p["output"]


class CostTracker:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.sessions: list[dict] = []

    def record(self, query: str, pruned_context: str, full_doc_tokens: int, output: str):
        pruned_tokens = count_tokens(pruned_context) + count_tokens(query)
        output_tokens = count_tokens(output)

        pruned_cost = estimate_cost(pruned_tokens, output_tokens, self.model)
        baseline_cost = estimate_cost(full_doc_tokens + count_tokens(query), output_tokens, self.model)

        entry = {
            "query_tokens": count_tokens(query),
            "pruned_context_tokens": count_tokens(pruned_context),
            "full_doc_tokens": full_doc_tokens,
            "output_tokens": output_tokens,
            "pruned_total_input": pruned_tokens,
            "baseline_total_input": full_doc_tokens + count_tokens(query),
            "pruned_cost_usd": pruned_cost,
            "baseline_cost_usd": baseline_cost,
            "savings_usd": baseline_cost - pruned_cost,
            "savings_pct": ((baseline_cost - pruned_cost) / baseline_cost * 100) if baseline_cost > 0 else 0,
        }
        self.sessions.append(entry)
        return entry

    def summary(self) -> dict:
        if not self.sessions:
            return {}
        total_pruned = sum(s["pruned_cost_usd"] for s in self.sessions)
        total_baseline = sum(s["baseline_cost_usd"] for s in self.sessions)
        return {
            "queries": len(self.sessions),
            "total_pruned_cost_usd": round(total_pruned, 6),
            "total_baseline_cost_usd": round(total_baseline, 6),
            "total_savings_usd": round(total_baseline - total_pruned, 6),
            "avg_savings_pct": round(
                sum(s["savings_pct"] for s in self.sessions) / len(self.sessions), 1
            ),
        }
