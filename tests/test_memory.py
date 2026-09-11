"""Tests for Task 8 session memory."""

import pytest

from agents.memory import (
    ask_with_memory,
    clear_all_sessions,
    clear_session,
    get_session_history,
    session_summary,
)


@pytest.fixture(autouse=True)
def clean_memory():
    clear_all_sessions()
    yield
    clear_all_sessions()


def test_history_is_created_per_session():
    h1 = get_session_history("A")
    h2 = get_session_history("B")
    assert h1 is not h2
    assert h1.messages == []
    assert h2.messages == []


def test_memory_chain_carries_history_between_turns(monkeypatch):
    calls = []

    def fake_run_crew(request):
        calls.append(request)
        return "MOCK CREW RESPONSE"

    monkeypatch.setattr("agents.crew.run_crew", fake_run_crew)

    first = ask_with_memory("session-1", "What is the status of order NYK-0028?")
    second = ask_with_memory("session-1", "What did I ask you about earlier?")

    assert first["input"] == "What is the status of order NYK-0028?"
    assert "no previous conversation" in calls[0]

    assert "What is the status of order NYK-0028?" in second["request_sent_to_crew"]
    summary = session_summary("session-1")
    assert summary["message_count"] == 4
    assert summary["messages"][0]["content"] == "What is the status of order NYK-0028?"
    assert summary["messages"][2]["content"] == "What did I ask you about earlier?"


def test_fresh_session_does_not_inherit_previous_state(monkeypatch):
    calls = []

    def fake_run_crew(request):
        calls.append(request)
        return "MOCK CREW RESPONSE"

    monkeypatch.setattr("agents.crew.run_crew", fake_run_crew)

    ask_with_memory("session-A", "What is the status of order NYK-0028?")
    ask_with_memory("session-B", "What did I ask you about earlier?")

    assert "NYK-0028" in calls[0]
    assert "no previous conversation" in calls[1]
    assert "NYK-0028" not in calls[1]


def test_clear_session_resets_only_that_session(monkeypatch):
    monkeypatch.setattr("agents.crew.run_crew", lambda request: "OK")

    ask_with_memory("A", "first")
    ask_with_memory("B", "second")
    clear_session("A")

    assert session_summary("A")["message_count"] == 0
    assert session_summary("B")["message_count"] == 2
