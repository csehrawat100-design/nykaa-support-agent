import pytest

from governance.authorization import (
    authorize_tool_call, can_call_tool, GovernanceAuthorizationError
)
from governance.risk import SYSTEM_RISK_LEVEL, get_risk_assessment
from governance.budget import RequestBudget, BudgetExceededError, enforce_budget


def test_only_lookup_agent_can_call_order_tool():
    assert authorize_tool_call("Lookup Agent", "check_order_status") is True
    assert can_call_tool("Retrieval Agent", "check_order_status") is False
    assert can_call_tool("Response Composer", "check_order_status") is False
    with pytest.raises(GovernanceAuthorizationError):
        authorize_tool_call("Retrieval Agent", "check_order_status")
    with pytest.raises(GovernanceAuthorizationError):
        authorize_tool_call("Response Composer", "check_order_status")


def test_unknown_tool_is_blocked_for_all_agents():
    with pytest.raises(GovernanceAuthorizationError):
        authorize_tool_call("Lookup Agent", "unknown_tool")


def test_risk_is_medium_with_justification():
    assessment = get_risk_assessment()
    assert SYSTEM_RISK_LEVEL == "Medium"
    assert assessment.level == "Medium"
    assert "customer-support" in assessment.rationale
    assert assessment.mitigations


def test_normal_request_is_within_budget():
    result = enforce_budget(300, 150)
    assert result["accepted"] is True
    assert result["estimated_cost_inr"] <= result["max_cost_inr"]


def test_deliberately_oversized_request_is_rejected():
    with pytest.raises(BudgetExceededError):
        enforce_budget(5000, 5000)


def test_budget_policy_can_be_customized():
    budget = RequestBudget(max_input_tokens=100, max_output_tokens=50, max_cost_inr=0.10)
    with pytest.raises(BudgetExceededError):
        enforce_budget(101, 10, budget)
