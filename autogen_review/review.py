from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .mock_model_client import MockReviewModelClient
from .schemas import ReviewVerdict


@dataclass(frozen=True)
class ReviewInput:
    query: str
    draft_answer: str
    retrieved_context: str


def _build_task(item: ReviewInput) -> str:
    return f"""Review this CrewAI draft before it reaches the customer.

ORIGINAL CUSTOMER QUERY:
{item.query}

CREWAI DRAFT:
{item.draft_answer}

ORIGINAL RETRIEVED CONTEXT:
{item.retrieved_context}

The Policy Compliance Reviewer must identify unsupported claims.
The Final Editor must approve the draft unchanged when it is grounded.
If it contains an unsupported claim, revise it so the unsupported claim is
removed. The final verdict must be structured JSON matching ReviewVerdict.
"""


def _build_team():
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.messages import StructuredMessage
    from autogen_agentchat.teams import RoundRobinGroupChat

    reviewer = AssistantAgent(
        name="Policy_Compliance_Reviewer",
        model_client=MockReviewModelClient(),
        system_message=(
            "You are the Policy Compliance Reviewer. Compare the CrewAI draft "
            "only against the supplied retrieved context. Identify unsupported "
            "claims and tell the Final Editor what must be corrected."
        ),
    )

    editor = AssistantAgent(
        name="Final_Editor",
        model_client=MockReviewModelClient(),
        system_message=(
            "You are the Final Editor. Use the original query, CrewAI draft, "
            "retrieved context, and reviewer message. Approve a grounded draft "
            "unchanged. If an unsupported claim exists, remove it. Return only "
            "a structured ReviewVerdict."
        ),
        output_content_type=ReviewVerdict,
    )

    return RoundRobinGroupChat(
        [reviewer, editor],
        termination_condition=MaxMessageTermination(3),
        max_turns=2,
        custom_message_types=[StructuredMessage[ReviewVerdict]],
    )


async def _review_async(item: ReviewInput) -> ReviewVerdict:
    team = _build_team()
    result = await team.run(task=_build_task(item))

    for message in reversed(result.messages):
        content = getattr(message, "content", None)
        if isinstance(content, ReviewVerdict):
            return content

    raise RuntimeError("AutoGen review completed without a structured ReviewVerdict.")


def review_draft(
    *,
    query: str,
    draft_answer: str,
    retrieved_context: str,
) -> ReviewVerdict:
    return asyncio.run(
        _review_async(
            ReviewInput(
                query=query,
                draft_answer=draft_answer,
                retrieved_context=retrieved_context,
            )
        )
    )
