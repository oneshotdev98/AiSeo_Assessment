"""Pydantic request/response models for both endpoints.

Two model families live here:
- Public API contracts (request/response bodies).
- Internal LLM payload models used to validate what the model returns
  before it ever reaches the caller.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class APIError(BaseModel):
    """Error envelope used by all non-2xx responses."""

    error: str
    message: str
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Feature 1 — AEO Content Scorer
# ---------------------------------------------------------------------------

class InputType(str, Enum):
    url = "url"
    text = "text"


class AEOAnalyzeRequest(BaseModel):
    input_type: InputType
    input_value: str = Field(min_length=1, description="A URL or raw HTML/plain text")

    @field_validator("input_value")
    @classmethod
    def strip_value(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("input_value must not be empty")
        return v


class CheckResult(BaseModel):
    check_id: str
    name: str
    passed: bool
    score: int = Field(ge=0)
    max_score: int = Field(ge=1)
    details: dict[str, Any]
    recommendation: Optional[str] = None


class AEOAnalyzeResponse(BaseModel):
    aeo_score: int = Field(ge=0, le=100)
    band: str
    checks: list[CheckResult]


# ---------------------------------------------------------------------------
# Feature 2 — Query Fan-Out Engine
# ---------------------------------------------------------------------------

class SubQueryType(str, Enum):
    comparative = "comparative"
    feature_specific = "feature_specific"
    use_case = "use_case"
    trust_signals = "trust_signals"
    how_to = "how_to"
    definitional = "definitional"


class FanoutRequest(BaseModel):
    target_query: str = Field(min_length=3, max_length=500)
    existing_content: Optional[str] = Field(
        default=None,
        description="Optional article text; when present, gap analysis runs.",
    )


class SubQueryResult(BaseModel):
    type: SubQueryType
    query: str
    # Only populated when existing_content was provided; the route uses
    # response_model_exclude_none so these are omitted otherwise.
    covered: Optional[bool] = None
    similarity_score: Optional[float] = None


class GapSummary(BaseModel):
    covered: int
    total: int
    coverage_percent: int
    covered_types: list[SubQueryType]
    missing_types: list[SubQueryType]


class FanoutResponse(BaseModel):
    target_query: str
    model_used: str
    total_sub_queries: int
    sub_queries: list[SubQueryResult]
    gap_summary: Optional[GapSummary] = None


# ---------------------------------------------------------------------------
# Internal — what we accept from the LLM
# ---------------------------------------------------------------------------

class LLMSubQuery(BaseModel):
    """One sub-query as returned by the LLM.

    extra="ignore" silently drops hallucinated fields instead of failing the
    whole payload; the enum on `type` rejects invented query types.
    """

    model_config = ConfigDict(extra="ignore")

    type: SubQueryType
    query: str = Field(min_length=5, max_length=300)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()
