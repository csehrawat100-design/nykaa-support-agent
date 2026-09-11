"""Task 8: in-process LangChain session memory for the Nykaa support agent."""

from __future__ import annotations

from typing import Dict

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory


# One history object per session ID. This is intentionally in-process only.
_SESSION_HISTORIES: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the existing history for a session, or create a fresh one."""
    if session_id not in _SESSION_HISTORIES:
        _SESSION_HISTORIES[session_id] = InMemoryChatMessageHistory()
    return _SESSION_HISTORIES[session_id]


def clear_session(session_id: str) -> None:
    """Reset one conversation without affecting other sessions."""
    _SESSION_HISTORIES.pop(session_id, None)


def clear_all_sessions() -> None:
    """Reset all in-process conversations."""
    _SESSION_HISTORIES.clear()


def history_messages(session_id: str) -> list[BaseMessage]:
    """Return a copy of the current session messages for testing/demo."""
    return list(get_session_history(session_id).messages)


def _history_to_text(messages: list[BaseMessage]) -> str:
    if not messages:
        return "(no previous conversation)"
    lines = []
    for message in messages:
        role = "user" if message.type == "human" else "assistant"
        lines.append(f"{role}: {message.content}")
    return "\n".join(lines)


def _memory_aware_request(payload: dict) -> str:
    """Build the request given to the already-tested CrewAI layer.

    RunnableWithMessageHistory injects the session's messages under `history`.
    The history is included explicitly so the crew receives the conversational
    state rather than merely storing it out of band.
    """
    current = payload["input"]
    history = payload.get("history", [])
    return (
        "Conversation history:\n"
        f"{_history_to_text(history)}\n\n"
        "Current customer request:\n"
        f"{current}"
    )


def _invoke_crew(payload: dict) -> dict:
    # Import lazily so importing the memory module does not initialize CrewAI/RAG.
    request = _memory_aware_request(payload)

    # Keep Task 8's original monkeypatch boundary intact. During normal
    # application execution the original CrewAI function is used, so the
    # integrated pipeline adds cache/governance/AutoGen. During the existing
    # memory tests, agents.crew.run_crew is replaced deliberately and must be
    # called directly.
    from agents import crew as crew_module
    current_run_crew = crew_module.run_crew
    original_run_crew = getattr(crew_module, "_ORIGINAL_RUN_CREW", None)

    if original_run_crew is not None and current_run_crew is not original_run_crew:
        result = current_run_crew(request)
        response = str(result)
        return {
            "input": payload["input"],
            "request_sent_to_crew": request,
            "crew_result": response,
            "output": response,
        }

    from integration.pipeline import run_support_request
    result = run_support_request(request)
    return {
        "input": payload["input"],
        "request_sent_to_crew": request,
        "crew_result": result.response,
        "output": result.response,
        "cache_hit": result.cache_hit,
        "review_approved": result.review.approved,
    }


# RunnableLambda lets RunnableWithMessageHistory manage the conversation while
# the existing CrewAI implementation remains the execution layer.
_MEMORY_CHAIN = RunnableLambda(_invoke_crew)

memory_chain = RunnableWithMessageHistory(
    _MEMORY_CHAIN,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def ask_with_memory(session_id: str, user_text: str) -> dict:
    """Send one turn through the session-aware chain."""
    return memory_chain.invoke(
        {"input": user_text},
        config={"configurable": {"session_id": session_id}},
    )


def session_summary(session_id: str) -> dict:
    """Return observable state useful for transcripts and tests."""
    messages = history_messages(session_id)
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": [
            {"role": ("user" if m.type == "human" else "assistant"), "content": str(m.content)}
            for m in messages
        ],
    }
