"""Pydantic response schemas for the Nykaa support agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ResponseFormat(BaseModel):
    """The only response shape accepted from the CrewAI composer."""

    model_config = ConfigDict(extra="forbid")

    response: str = Field(
        ...,
        min_length=1,
        description="Customer-facing support answer.",
    )
    source_type: Literal["policy", "order", "mixed", "fallback"] = Field(
        ...,
        description="Primary source used for the answer.",
    )
    grounded: bool = Field(
        ...,
        description="Whether the answer is explicitly grounded in the supplied evidence.",
    )