"""
Nykaa Domain Support Agent
Part 1 - Task 1: Deterministic Order Dataset Generator

Generates a reproducible synthetic order dataset satisfying the capstone
requirements:

- At least 40 order records
- All required categories represented
- All required statuses represented
- Every required category has at least 3 records
- days_since_created is between 0 and 30
- delayed_shipment percentage is between 10% and 30%
- Deterministic generation using a fixed random seed

Design choices:
- Seed: 20260000
- Number of records: 50
- Order value range: ₹499–₹24,999
- Category weights: Apparel 24, Electronics 20, Home 18,
  Footwear 20, Beauty 18
- Status weights: Placed 20, Shipped 25, Delivered 35,
  Returned 10, Refunded 10
- Delayed shipment probability: 20%

The price range is intended to cover realistic synthetic e-commerce
orders from lower-priced beauty/apparel purchases through higher-value
electronics and home purchases.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Dataset design
# ---------------------------------------------------------------------------

SEED = 20260000
NUM_RECORDS = 50

MIN_ORDER_VALUE_INR = 499
MAX_ORDER_VALUE_INR = 24_999

CATEGORIES = [
    "Apparel",
    "Electronics",
    "Home",
    "Footwear",
    "Beauty",
]

CATEGORY_WEIGHTS = {
    "Apparel": 24,
    "Electronics": 20,
    "Home": 18,
    "Footwear": 20,
    "Beauty": 18,
}

STATUSES = [
    "Placed",
    "Shipped",
    "Delivered",
    "Returned",
    "Refunded",
]

STATUS_WEIGHTS = {
    "Placed": 20,
    "Shipped": 25,
    "Delivered": 35,
    "Returned": 10,
    "Refunded": 10,
}

DELAYED_SHIPMENT_PROBABILITY = 0.20


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def generate_orders(
    seed: int = SEED,
    num_records: int = NUM_RECORDS,
) -> list[dict[str, Any]]:
    """
    Generate a deterministic list of synthetic Nykaa order records.

    The function deliberately uses a local random.Random instance rather
    than the global random generator so that other modules cannot change
    the generated dataset accidentally.
    """

    rng = random.Random(seed)

    categories = list(CATEGORY_WEIGHTS.keys())
    category_weights = list(CATEGORY_WEIGHTS.values())

    statuses = list(STATUS_WEIGHTS.keys())
    status_weights = list(STATUS_WEIGHTS.values())

    orders: list[dict[str, Any]] = []

    for index in range(1, num_records + 1):
        category = rng.choices(
            categories,
            weights=category_weights,
            k=1,
        )[0]

        status = rng.choices(
            statuses,
            weights=status_weights,
            k=1,
        )[0]

        order_value = rng.randint(
            MIN_ORDER_VALUE_INR,
            MAX_ORDER_VALUE_INR,
        )

        days_since_created = rng.randint(0, 30)

        delayed_shipment = (
            rng.random() < DELAYED_SHIPMENT_PROBABILITY
        )

        order = {
            "record_id": f"NYK-{index:04d}",
            "category": category,
            "status": status,
            "order_value_inr": order_value,
            "days_since_created": days_since_created,
            "delayed_shipment": delayed_shipment,
        }

        orders.append(order)

    return orders


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_orders(orders: list[dict[str, Any]]) -> None:
    """
    Validate all structural requirements from Part 1 Task 1.

    Raises AssertionError if any requirement is violated.
    """

    # Minimum dataset size
    assert len(orders) >= 40, (
        f"Dataset must contain at least 40 records; "
        f"got {len(orders)}."
    )

    # Required fields
    required_fields = {
        "record_id",
        "category",
        "status",
        "order_value_inr",
        "days_since_created",
        "delayed_shipment",
    }

    for order in orders:
        assert required_fields.issubset(order.keys()), (
            f"Missing required field in order {order.get('record_id')}"
        )

        # record ID
        assert isinstance(order["record_id"], str)
        assert order["record_id"].startswith("NYK-")

        # Category
        assert order["category"] in CATEGORIES, (
            f"Unexpected category: {order['category']}"
        )

        # Status
        assert order["status"] in STATUSES, (
            f"Unexpected status: {order['status']}"
        )

        # Order value
        assert isinstance(order["order_value_inr"], int)
        assert MIN_ORDER_VALUE_INR <= order["order_value_inr"] <= MAX_ORDER_VALUE_INR

        # Days since created
        assert isinstance(order["days_since_created"], int)
        assert 0 <= order["days_since_created"] <= 30

        # Delayed shipment
        assert isinstance(order["delayed_shipment"], bool)

    # Every required category must occur at least once
    category_counts = Counter(order["category"] for order in orders)

    for category in CATEGORIES:
        assert category_counts[category] >= 1, (
            f"Required category missing: {category}"
        )

    # Every required category must have at least 3 records
    for category in CATEGORIES:
        assert category_counts[category] >= 3, (
            f"Category {category} has only "
            f"{category_counts[category]} records; "
            f"minimum is 3."
        )

    # Every required status must occur at least once
    status_counts = Counter(order["status"] for order in orders)

    for status in STATUSES:
        assert status_counts[status] >= 1, (
            f"Required status missing: {status}"
        )

    # Delayed shipment percentage
    delayed_count = sum(
        order["delayed_shipment"]
        for order in orders
    )

    delayed_percentage = (
        delayed_count / len(orders)
    ) * 100

    assert 10 <= delayed_percentage <= 30, (
        f"Delayed shipment percentage must be between "
        f"10% and 30%; got {delayed_percentage:.2f}%."
    )

    # Record IDs should be unique
    record_ids = [order["record_id"] for order in orders]

    assert len(record_ids) == len(set(record_ids)), (
        "Duplicate record_id values detected."
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(orders: list[dict[str, Any]]) -> None:
    """
    Print the dataset statistics required by the capstone.
    """

    category_counts = Counter(
        order["category"]
        for order in orders
    )

    status_counts = Counter(
        order["status"]
        for order in orders
    )

    delayed_count = sum(
        order["delayed_shipment"]
        for order in orders
    )

    delayed_percentage = (
        delayed_count / len(orders)
    ) * 100

    print("=" * 70)
    print("NYKAA DOMAIN SUPPORT AGENT - DATASET REPORT")
    print("=" * 70)

    print(f"Seed: {SEED}")
    print(f"Number of records: {len(orders)}")
    print(
        f"Order value range: "
        f"₹{MIN_ORDER_VALUE_INR:,} - ₹{MAX_ORDER_VALUE_INR:,}"
    )

    print("\nCategory counts:")
    for category in CATEGORIES:
        print(
            f"  {category:<15} "
            f"{category_counts[category]:>3}"
        )

    print("\nStatus counts:")
    for status in STATUSES:
        print(
            f"  {status:<15} "
            f"{status_counts[status]:>3}"
        )

    print("\nDelayed shipments:")
    print(f"  Delayed records : {delayed_count}")
    print(f"  Total records   : {len(orders)}")
    print(f"  Percentage      : {delayed_percentage:.2f}%")

    print("\nValidation:")
    print("  ✓ Minimum 40 records")
    print("  ✓ All required categories present")
    print("  ✓ Every category has at least 3 records")
    print("  ✓ All required statuses present")
    print("  ✓ days_since_created in range 0-30")
    print("  ✓ delayed_shipment percentage in range 10%-30%")
    print("  ✓ Unique record IDs")

    print("=" * 70)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def save_orders(
    orders: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    """
    Save generated orders to dataset/orders.json.
    """

    if output_path is None:
        output_path = Path(__file__).parent / "orders.json"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            orders,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ORDERS = generate_orders()

validate_orders(ORDERS)


if __name__ == "__main__":
    print_report(ORDERS)

    output_file = save_orders(ORDERS)

    print(f"\nDataset saved to: {output_file}")