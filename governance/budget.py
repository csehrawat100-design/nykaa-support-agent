"""Runtime-layer per-request token/cost budget enforcement.

This is a deterministic pre-flight guard for the local MOCK_LLM path. It uses
an explicit token estimate and a simple configured unit cost so oversized
requests are rejected before execution.
"""
from dataclasses import dataclass

class BudgetExceededError(RuntimeError):
    """Raised when a request exceeds the configured runtime budget."""

@dataclass(frozen=True)
class RequestBudget:
    max_input_tokens: int = 1200
    max_output_tokens: int = 500
    max_cost_inr: float = 0.50
    input_cost_per_token_inr: float = 0.0002
    output_cost_per_token_inr: float = 0.0004

    def estimate_cost_inr(self, input_tokens: int, output_tokens: int) -> float:
        return round(
            input_tokens * self.input_cost_per_token_inr
            + output_tokens * self.output_cost_per_token_inr, 6
        )


def enforce_budget(
    input_tokens: int,
    output_tokens: int,
    budget: RequestBudget | None = None,
) -> dict:
    """Reject oversized requests and return deterministic budget evidence."""
    policy = budget or RequestBudget()
    cost = policy.estimate_cost_inr(input_tokens, output_tokens)
    violations = []
    if input_tokens > policy.max_input_tokens:
        violations.append(f"input_tokens {input_tokens} > {policy.max_input_tokens}")
    if output_tokens > policy.max_output_tokens:
        violations.append(f"output_tokens {output_tokens} > {policy.max_output_tokens}")
    if cost > policy.max_cost_inr:
        violations.append(f"estimated_cost_inr {cost} > {policy.max_cost_inr}")
    if violations:
        raise BudgetExceededError("Runtime budget exceeded: " + "; ".join(violations))
    return {
        "accepted": True,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_inr": cost,
        "max_input_tokens": policy.max_input_tokens,
        "max_output_tokens": policy.max_output_tokens,
        "max_cost_inr": policy.max_cost_inr,
    }
