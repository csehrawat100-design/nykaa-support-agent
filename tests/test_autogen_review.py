from autogen_review.review import _build_team, review_draft
from autogen_review.schemas import ReviewVerdict


def test_clean_draft_is_approved_unchanged():
    verdict = review_draft(
        query="What is the return window for beauty products?",
        draft_answer="The return window is stated in the supplied policy.",
        retrieved_context=(
            "return_window: Beauty returns follow the documented return window."
        ),
    )

    assert isinstance(verdict, ReviewVerdict)
    assert verdict.approved is True
    assert verdict.final_answer == (
        "The return window is stated in the supplied policy."
    )


def test_deliberately_ungrounded_draft_is_revised():
    verdict = review_draft(
        query="What is the return window for beauty products?",
        draft_answer=(
            "DELIBERATE_UNGROUNDED_CLAIM: "
            "Beauty refunds are always instant and guaranteed."
        ),
        retrieved_context=(
            "return_window: Beauty returns follow the documented return window."
        ),
    )

    assert isinstance(verdict, ReviewVerdict)
    assert verdict.approved is False
    assert "always instant and guaranteed" not in verdict.final_answer
    assert "unsupported claim" in verdict.reason


def test_review_team_is_two_agent_round_robin():
    from autogen_agentchat.teams import RoundRobinGroupChat

    team = _build_team()
    assert isinstance(team, RoundRobinGroupChat)
    assert len(team._participants) == 2

    # AutoGen 0.7.5 stores the configured value privately.
    assert team._termination_condition._max_messages == 3
