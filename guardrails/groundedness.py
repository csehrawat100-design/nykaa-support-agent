"""Output groundedness guardrail using Task 4's calibrated threshold."""
def check_groundedness(retrieved_results, threshold: float = 0.4844) -> dict:
    rows = list(retrieved_results)
    scores = []
    for row in rows:
        try:
            scores.append(float(row.get("similarity", row.get("score", 0.0))))
        except (TypeError, ValueError):
            pass
    maximum = max(scores, default=0.0)
    allowed = maximum >= threshold
    return {"allowed": allowed, "max_similarity": maximum,
            "reason": ("Retrieved context clears the calibrated threshold."
                       if allowed else
                       f"Retrieved evidence is below the calibrated threshold {threshold:.4f}.")}

def enforce_groundedness(retrieved_results, threshold: float = 0.4844) -> dict:
    result = check_groundedness(retrieved_results, threshold)
    if not result["allowed"]:
        raise ValueError("Groundedness guardrail blocked the response: " + result["reason"])
    return result