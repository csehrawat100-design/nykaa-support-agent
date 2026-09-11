from pydantic import BaseModel, ConfigDict, Field


class ReviewVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool = Field(description="Whether the draft is approved unchanged.")
    final_answer: str = Field(description="Customer-facing answer after review.")
    reason: str = Field(description="Why the draft was approved or revised.")
