import pytest

from integration.pipeline import clear_cache, cache_stats
from governance.authorization import GovernanceAuthorizationError, authorize_tool_call


def setup_function():
    clear_cache()


def test_pipeline_has_real_cache_hit(monkeypatch):
    import integration.pipeline as pipeline

    calls = {"crew": 0, "review": 0}

    class Draft:
        response = "The return window is stated in the supplied policy."
        source_type = "policy"
        grounded = True

    class Verdict:
        approved = True
        final_answer = "The return window is stated in the supplied policy."
        reason = "Grounded."

        def model_dump(self):
            return {"approved": self.approved, "final_answer": self.final_answer, "reason": self.reason}

    monkeypatch.setattr(pipeline, "run_crew", lambda q: calls.__setitem__("crew", calls["crew"] + 1) or Draft())
    monkeypatch.setattr(pipeline, "review_draft", lambda **kwargs: calls.__setitem__("review", calls["review"] + 1) or Verdict())
    monkeypatch.setattr(pipeline, "_retrieved_context_for_review", lambda q: "return_window: documented policy")

    first = pipeline.run_support_request("What is the return window?")
    second = pipeline.run_support_request("  WHAT   IS the return window?  ")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls == {"crew": 1, "review": 1}
    assert cache_stats()["calls_avoided"] == 1


def test_lookup_authorization_is_enforced_at_tool_boundary():
    assert authorize_tool_call("Lookup Agent", "check_order_status") is True
    with pytest.raises(GovernanceAuthorizationError):
        authorize_tool_call("Retrieval Agent", "check_order_status")
    with pytest.raises(GovernanceAuthorizationError):
        authorize_tool_call("Response Composer", "check_order_status")
