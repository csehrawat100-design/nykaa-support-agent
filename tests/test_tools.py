"""Tests for the Task 6 order lookup tool."""

import pytest

from agents.tools import (
    ESCALATION_THRESHOLD,
    calculate_escalation_score,
    check_order_status,
)


def test_lookup_returns_required_fields():
    result = check_order_status("NYK-0001")

    assert result["record_id"] == "NYK-0001"
    assert result["status"] == "Delivered"
    assert result["order_value_inr"] == 7577
    assert 0.0 <= result["escalation_score"] <= 1.0


def test_score_formula():
    # NYK-0006: delayed=False, days_since_created=12
    expected = round(0.60 * 0.0 + 0.40 * (12 / 30), 4)

    result = check_order_status("NYK-0006")

    assert result["escalation_score"] == expected


def test_delayed_order_score():
    # NYK-0007: delayed=True, days_since_created=18
    expected = round(0.60 * 1.0 + 0.40 * (18 / 30), 4)

    result = check_order_status("NYK-0007")

    assert result["escalation_score"] == expected
    assert result["escalation_score"] > ESCALATION_THRESHOLD
    assert result["recommend_escalation"] is True


def test_score_is_bounded():
    assert calculate_escalation_score(
        delayed_shipment=False,
        days_since_created=0,
    ) == 0.0

    assert calculate_escalation_score(
        delayed_shipment=True,
        days_since_created=30,
    ) == 1.0


def test_unknown_record_raises():
    with pytest.raises(ValueError, match="not found"):
        check_order_status("NYK-9999")


def test_blank_record_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        check_order_status("   ")