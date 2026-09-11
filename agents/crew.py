"""Task 7: CrewAI multi-agent orchestration for the Nykaa support agent."""

from __future__ import annotations

import json
import os

from typing import Any, Dict, Literal

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

from .mock_llm import MockLLM
from .tools import check_order_status
from .schemas import ResponseFormat


# The selected configuration from Tasks 3–5 is intentionally fixed here.
SELECTED_CHUNK_STRATEGY = "sentence"
CALIBRATED_THRESHOLD = 0.4844


class RAGToolInput(BaseModel):
    query: str = Field(..., description="Customer policy question.")


class LookupToolInput(BaseModel):
    record_id: str = Field(..., pattern=r"^NYK-\d{4}$")


class RAGLookupTool(BaseTool):
    name: str = "rag_policy_search"
    description: str = (
        "Retrieve policy context from the selected sentence-based RAG collection. "
        "Use the calibrated similarity threshold 0.4844."
    )
    args_schema: type[BaseModel] = RAGToolInput

    def _run(self, query: str) -> str:
        from rag.retrieval import Retriever

        retriever = Retriever(
            strategy=SELECTED_CHUNK_STRATEGY,
            top_k=5,
        )

        results = retriever.retrieve(
            query=query,
            top_k=5,
        )

        relevant = [
            chunk
            for chunk in results
            if chunk.similarity >= CALIBRATED_THRESHOLD
        ]

        if not relevant:
            return (
                "No relevant policy context was retrieved above the "
                f"calibrated threshold of {CALIBRATED_THRESHOLD:.4f}."
            )

        return str(
            [
                {
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "similarity": round(chunk.similarity, 4),
                    "chunk_strategy": chunk.chunk_strategy,
                }
                for chunk in relevant
            ]
        )


class OrderLookupTool(BaseTool):
    name: str = "order_status_lookup"
    description: str = (
        "Look up one synthetic Nykaa order by record_id and return status, "
        "order value, escalation score and recommendation."
    )
    args_schema: type[BaseModel] = LookupToolInput

    def _run(self, record_id: str) -> str:
        return str(check_order_status(record_id))

def build_crew() -> Crew:
    """Construct the three-agent crew using deterministic MOCK_LLM."""

    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")

    llm = MockLLM()

    retrieval_agent = Agent(
        role="Retrieval Agent",
        goal="Retrieve only relevant policy context for the customer's question.",
        backstory=(
            "You are a policy retrieval specialist. Use only the provided RAG "
            "tool and the selected sentence collection."
        ),
        tools=[RAGLookupTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    lookup_agent = Agent(
        role="Lookup Agent",
        goal="Retrieve the status of a specific synthetic Nykaa order.",
        backstory=(
            "You are the only agent authorized to call the order lookup tool."
        ),
        tools=[OrderLookupTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    composer = Agent(
        role="Response Composer",
        goal="Combine the retrieval or lookup result into one concise supported draft.",
        backstory=(
            "You are a support response editor. Do not invent facts and do not "
            "call the order lookup tool."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    retrieval_task = Task(
        description=(
            "For a policy question, use the RAG tool and return the retrieved "
            "context and source metadata. Customer request: {request}"
        ),
        expected_output="Retrieved policy context with source information.",
        agent=retrieval_agent,
    )

    lookup_task = Task(
        description=(
            "If the request contains an order ID, use the order status tool. "
            "Otherwise state that no order lookup is needed. Customer request: {request}"
        ),
        expected_output="Order lookup result or a clear no-lookup statement.",
        agent=lookup_agent,
    )

    compose_task = Task(
        description=(
            "Combine the previous task outputs into one grounded support draft. "
            "Do not introduce facts absent from those outputs. Customer request: {request}"
        ),
        expected_output="A concise final support draft.",
        agent=composer,
        context=[retrieval_task, lookup_task],
    )

    return Crew(
        agents=[retrieval_agent, lookup_agent, composer],
        tasks=[retrieval_task, lookup_task, compose_task],
        process=Process.sequential,
        verbose=True,
    )

def _extract_response_format(result: Any) -> ResponseFormat:
    """Validate CrewAI output against the Pydantic response contract.

    CrewAI versions may expose the task result as `pydantic`, `json_dict`,
    or `raw`. Normalize those representations and validate them explicitly.
    """
    candidate = getattr(result, "pydantic", None)

    if isinstance(candidate, ResponseFormat):
        return ResponseFormat.model_validate(candidate.model_dump())

    json_dict = getattr(result, "json_dict", None)

    if isinstance(json_dict, dict):
        return ResponseFormat.model_validate(json_dict)

    raw = getattr(result, "raw", result)

    if isinstance(raw, ResponseFormat):
        return ResponseFormat.model_validate(raw.model_dump())

    if isinstance(raw, dict):
        return ResponseFormat.model_validate(raw)

    text = str(raw).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "CrewAI composer did not return valid ResponseFormat JSON."
        ) from exc

    return ResponseFormat.model_validate(parsed)

def validate_crew_response(result: Any) -> ResponseFormat:
    """Public validation boundary used by tests and later API layers."""
    try:
        return _extract_response_format(result)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValueError(f"Invalid structured crew response: {exc}") from exc

def run_crew(request: str) -> ResponseFormat:
    """Run the crew and return validated structured output."""
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["OTEL_SDK_DISABLED"] = "true"

    result = build_crew().kickoff(
        inputs={"request": request}
    )

    return validate_crew_response(result)


_ORIGINAL_RUN_CREW = run_crew

if __name__ == "__main__":
    print(run_crew("What is the return window for Beauty products?"))
    print(run_crew("What is the status of order NYK-0028?"))
