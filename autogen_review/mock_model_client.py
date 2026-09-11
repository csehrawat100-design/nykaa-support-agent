"""Deterministic zero-network AutoGen model client for Task 14.

This is an AutoGen ChatCompletionClient implementation, not an external
interceptor. It lets the real AssistantAgent + RoundRobinGroupChat pipeline run
under the capstone's MOCK_LLM requirement.
"""

from __future__ import annotations

import re
import json
from typing import Any, AsyncGenerator, Sequence

from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    RequestUsage,
)


class MockReviewModelClient(ChatCompletionClient):
    """Local deterministic model client; never calls an external API."""

    def __init__(self) -> None:
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._capabilities: ModelCapabilities = {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "structured_output": True,
        }

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "structured_output": True,
            "family": "mock",
        }

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _make_content(self, prompt: str) -> str:
        bad = "DELIBERATE_UNGROUNDED_CLAIM" in prompt

        # The reviewer should communicate a review note, not ReviewVerdict JSON.
        # The Final Editor is the only agent configured for structured output.
        is_final_editor = "You are the Final Editor" in prompt

        if not is_final_editor:
            if bad:
                return (
                    "REVISION_REQUIRED: The draft contains an unsupported claim "
                    "about instant and guaranteed beauty refunds. Remove that claim."
                )
            return (
                "APPROVE: The draft is consistent with the supplied context "
                "and contains no unsupported claim."
            )

        if bad:
            payload = {
                "approved": False,
                "final_answer": (
                    "The supplied context does not state that beauty refunds are "
                    "instant or guaranteed."
                ),
                "reason": (
                    "The draft contains an unsupported claim, so that claim "
                    "was removed."
                ),
            }
        else:
            draft_match = re.search(
                r"CREWAI DRAFT:\s*(.*?)\s*ORIGINAL RETRIEVED CONTEXT:",
                prompt,
                re.DOTALL,
            )

            draft = (
                draft_match.group(1).strip()
                if draft_match
                else "The supplied context supports the CrewAI draft."
            )

            payload = {
                "approved": True,
                "final_answer": draft,
                "reason": "The draft contains no unsupported claim.",
            }

        return json.dumps(payload, ensure_ascii=False)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] | None = None,
        json_output: bool | None = None,
        extra_create_args: dict[str, Any] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        prompt = "\n".join(
            str(getattr(message, "content", "")) for message in messages
        )

        print("\n========== MOCK REVIEW DEBUG ==========")
        print(prompt)
        print("=======================================\n")

        content = self._make_content(prompt)

        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(content) // 4)
        usage = RequestUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + completion_tokens,
        )

        return CreateResult(
            finish_reason="stop",
            content=content,
            usage=usage,
            cached=False,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] | None = None,
        json_output: bool | None = None,
        extra_create_args: dict[str, Any] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[Any, None]:
        yield await self.create(
            messages,
            tools=tools,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._total_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] | None = None,
    ) -> int:
        return sum(
            max(1, len(str(getattr(message, "content", ""))) // 4)
            for message in messages
        )

    def remaining_tokens(self, messages: Sequence[LLMMessage]) -> int:
        return max(0, 100_000 - self.count_tokens(messages))
