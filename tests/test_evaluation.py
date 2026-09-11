from evaluation.judge import average_scores, build_judge_prompt, evaluate

def _queries():
    from evaluation.queries import FULL_EVALUATION_QUERIES, IN_SCOPE_QUERIES
    return FULL_EVALUATION_QUERIES, IN_SCOPE_QUERIES

def test_uses_canonical_15_query_set():
    full, _ = _queries()
    assert len(full) == 15

def test_covers_all_12_required_kb_topics():
    full, _ = _queries()
    expected = {
        "return_window", "cod_refund_timelines", "delivery_sla",
        "reverse_pickup", "warranty", "cancellation", "loyalty_points",
        "payment_failure", "size_exchange", "damaged_item_claims",
        "international_shipping", "customer_support_escalation",
    }
    assert {d for q in full for d in q.expected_documents} == expected

def test_has_at_least_two_oos_or_edge_cases():
    full, _ = _queries()
    assert sum(not q.expected_documents for q in full) >= 2

def test_task_4_5_shared_queries_are_unchanged():
    full, shared = _queries()
    assert [q.query for q in shared] == [q.query for q in full[:5]]

def test_judge_prompt_contains_four_dimensions():
    _, shared = _queries()
    prompt = build_judge_prompt(shared[0], "The answer is supported by policy.")
    for term in ("Accuracy", "Grounding", "Completeness", "Safety"):
        assert term in prompt

def test_oos_refusal_scores_safe():
    full, _ = _queries()
    oos = next(q for q in full if not q.expected_documents)
    result = evaluate(oos, "I can't verify that request from the Nykaa support knowledge base.")
    assert result["safety"] == 5
    assert result["accuracy"] == 5

def test_average_contains_four_metrics():
    full, _ = _queries()
    results = [evaluate(q, "I can't verify that from the supported knowledge base.") for q in full]
    averages = average_scores(results)
    assert set(averages) == {"accuracy", "grounding", "completeness", "safety"}
    assert all(1 <= value <= 5 for value in averages.values())
