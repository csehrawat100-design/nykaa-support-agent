"""Reusable evaluation query sets for the Nykaa Support Agent.

The first five in-scope queries are the shared Task 4 / Task 5 set.  The
15-query set covers every required knowledge-base topic and includes two
out-of-scope queries for the Part 3 evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    """A test query and its expected parent knowledge-base document(s)."""

    query: str
    expected_documents: tuple[str, ...]


# ---------------------------------------------------------------------------
# Task 4 + Task 5 shared evaluation set
# ---------------------------------------------------------------------------
# These five queries are intentionally stable. Task 5 must use the same
# in-scope queries used for Task 4.
IN_SCOPE_QUERIES: tuple[EvaluationQuery, ...] = (
    EvaluationQuery(
        query="What is the return window for beauty products?",
        expected_documents=("return_window",),
    ),
    EvaluationQuery(
        query="How long does a COD refund take after approval?",
        expected_documents=("cod_refund_timelines",),
    ),
    EvaluationQuery(
        query="How long does delivery usually take after order confirmation?",
        expected_documents=("delivery_sla",),
    ),
    EvaluationQuery(
        query="What are the requirements for a reverse pickup?",
        expected_documents=("reverse_pickup",),
    ),
    EvaluationQuery(
        query="What should I do if my order arrives damaged?",
        expected_documents=("damaged_item_claims",),
    ),
)


# ---------------------------------------------------------------------------
# Task 4 threshold calibration
# ---------------------------------------------------------------------------
# The guideline requires >=3 in-scope and >=2 deliberately out-of-scope
# queries. Three shared in-scope queries are reused to keep calibration and
# evaluation consistent.
CALIBRATION_IN_SCOPE_QUERIES: tuple[str, ...] = (
    IN_SCOPE_QUERIES[0].query,
    IN_SCOPE_QUERIES[1].query,
    IN_SCOPE_QUERIES[2].query,
)

CALIBRATION_OUT_OF_SCOPE_QUERIES: tuple[str, ...] = (
    "What is the current price of gold in India?",
    "Who won the latest Formula 1 race?",
)

# Query used to demonstrate the Task 4 out-of-scope fallback.
FALLBACK_QUERY = "What is the process for changing my Instagram password?"

# ---------------------------------------------------------------------------
# Part 3: 15-query evaluation set
# ---------------------------------------------------------------------------
# Covers all 12 required KB topics plus two deliberately out-of-scope queries.
# The five shared Task 4/5 queries are included unchanged.
FULL_EVALUATION_QUERIES: tuple[EvaluationQuery, ...] = (
    *IN_SCOPE_QUERIES,
    EvaluationQuery(
        query="What warranty coverage is available for electronics?",
        expected_documents=("warranty",),
    ),
    EvaluationQuery(
        query="Can I cancel my order after it has been shipped?",
        expected_documents=("cancellation",),
    ),
    EvaluationQuery(
        query="When can I redeem my loyalty points?",
        expected_documents=("loyalty_points",),
    ),
    EvaluationQuery(
        query="What should I do if my payment failed but I am not sure whether the order was created?",
        expected_documents=("payment_failure",),
    ),
    EvaluationQuery(
        query="Can I exchange the size of an eligible footwear item?",
        expected_documents=("size_exchange",),
    ),
    EvaluationQuery(
        query="Can Nykaa ship this product internationally and will customs charges apply?",
        expected_documents=("international_shipping",),
    ),
    EvaluationQuery(
        query="When should a delayed order or unresolved refund be escalated to support?",
        expected_documents=("customer_support_escalation",),
    ),
    EvaluationQuery(
        query="Can you tell me the current stock price of Nykaa?",
        expected_documents=(),
    ),
    EvaluationQuery(
        query="What is the weather forecast for Mumbai tomorrow?",
        expected_documents=(),
    ),
    # The following query is an edge case
    EvaluationQuery(
        query="Can you change my delivery address to a new address after dispatch?",
        expected_documents=(),
    ),
)


# Plain query strings are convenient for generation and threshold calibration.
IN_SCOPE_QUERY_TEXTS: tuple[str, ...] = tuple(
    item.query for item in IN_SCOPE_QUERIES
)

FULL_EVALUATION_QUERY_TEXTS: tuple[str, ...] = tuple(
    item.query for item in FULL_EVALUATION_QUERIES
)


# Mapping shape expected by rag.evaluation.evaluate_strategy().
IN_SCOPE_QUERY_MAPPING: dict[str, tuple[str, ...]] = {
    item.query: item.expected_documents for item in IN_SCOPE_QUERIES
}

FULL_EVALUATION_QUERY_MAPPING: dict[str, tuple[str, ...]] = {
    item.query: item.expected_documents for item in FULL_EVALUATION_QUERIES
}


__all__ = [
    "EvaluationQuery",
    "IN_SCOPE_QUERIES",
    "IN_SCOPE_QUERY_TEXTS",
    "IN_SCOPE_QUERY_MAPPING",
    "CALIBRATION_IN_SCOPE_QUERIES",
    "CALIBRATION_OUT_OF_SCOPE_QUERIES",
    "FALLBACK_QUERY",
    "FULL_EVALUATION_QUERIES",
    "FULL_EVALUATION_QUERY_TEXTS",
    "FULL_EVALUATION_QUERY_MAPPING",
]
