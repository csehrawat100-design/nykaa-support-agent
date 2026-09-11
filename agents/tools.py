"""Task 6 lookup tools for the Nykaa Support Agent.

The order lookup is deliberately deterministic and reads the synthetic dataset
created in Task 1. No network access or external service is used.
"""

from __future__ import annotations

from typing import Any

from dataset.dataset import ORDERS

MAX_DAYS_SINCE_CREATED = 30
DELAY_WEIGHT = 0.60
RECENCY_WEIGHT = 0.40
ESCALATION_THRESHOLD = 0.70


def _normalized_recency(days_since_created: int) -> float:
    """Normalize dataset recency from 0..30 days to a 0..1 signal."""
    if not 0 <= days_since_created <= MAX_DAYS_SINCE_CREATED:
        raise ValueError(
            f"days_since_created must be between 0 and {MAX_DAYS_SINCE_CREATED}."
        )
    return days_since_created / MAX_DAYS_SINCE_CREATED


def calculate_escalation_score(
    *, delayed_shipment: bool, days_since_created: int
) -> float:
    """Return a designed escalation score in the inclusive range [0, 1].

    Formula:
        score = 0.60 * delayed_shipment + 0.40 * (days_since_created / 30)

    Delayed shipment is the stronger signal because it directly indicates a
    fulfillment problem. Recency contributes a continuous secondary signal.
    """
    recency_signal = _normalized_recency(days_since_created)
    score = (
        DELAY_WEIGHT * float(delayed_shipment)
        + RECENCY_WEIGHT * recency_signal
    )
    return round(min(1.0, max(0.0, score)), 4)


def check_order_status(record_id: str) -> dict[str, Any]:
    """Look up an order and return status, value, and escalation score.

    Args:
        record_id: Synthetic order identifier such as ``NYK-0001``.

    Returns:
        A dictionary containing the requested order fields plus the designed
        escalation score and escalation recommendation.

    Raises:
        ValueError: If the record ID is blank or is not present in ORDERS.
    """
    record_id = record_id.strip()
    if not record_id:
        raise ValueError("record_id must not be empty.")

    order = next(
        (item for item in ORDERS if item["record_id"] == record_id),
        None,
    )

    if order is None:
        raise ValueError(f"Order '{record_id}' was not found.")

    escalation_score = calculate_escalation_score(
        delayed_shipment=order["delayed_shipment"],
        days_since_created=order["days_since_created"],
    )

    return {
        "record_id": order["record_id"],
        "status": order["status"],
        "order_value_inr": order["order_value_inr"],
        "escalation_score": escalation_score,
        "recommend_escalation": escalation_score > ESCALATION_THRESHOLD,
    }


__all__ = [
    "MAX_DAYS_SINCE_CREATED",
    "DELAY_WEIGHT",
    "RECENCY_WEIGHT",
    "ESCALATION_THRESHOLD",
    "calculate_escalation_score",
    "check_order_status",
]
