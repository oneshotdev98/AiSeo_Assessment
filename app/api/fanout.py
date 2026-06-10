"""POST /api/fanout/generate — LLM fan-out + optional semantic gap analysis.

The endpoint is async: the LLM call is pure I/O and awaits cleanly. The
embedding step is CPU-bound, so it is pushed onto the threadpool with
run_in_threadpool to keep the event loop responsive.
"""

from __future__ import annotations

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.models.schemas import (
    FanoutRequest,
    FanoutResponse,
    GapSummary,
    SubQueryResult,
    SubQueryType,
)
from app.services.fanout_engine import generate_sub_queries
from app.services.gap_analyzer import get_similarity_threshold, max_similarities

router = APIRouter()


@router.post("/generate", response_model=FanoutResponse, response_model_exclude_none=True)
async def generate(request: FanoutRequest) -> FanoutResponse:
    sub_queries, model_name = await generate_sub_queries(request.target_query)

    results = [SubQueryResult(type=sq.type, query=sq.query) for sq in sub_queries]
    gap_summary = None

    content = (request.existing_content or "").strip()
    if content:
        threshold = get_similarity_threshold()
        similarities = await run_in_threadpool(
            max_similarities, [sq.query for sq in sub_queries], content
        )
        for result, similarity in zip(results, similarities):
            result.similarity_score = similarity
            result.covered = similarity >= threshold

        covered_count = sum(1 for r in results if r.covered)
        # A type counts as covered only when ALL of its sub-queries are
        # covered — a single gap inside a type is still actionable work.
        types_present = {r.type for r in results}
        covered_types = sorted(
            (t for t in types_present if all(r.covered for r in results if r.type == t)),
            key=lambda t: t.value,
        )
        missing_types = sorted(types_present - set(covered_types), key=lambda t: t.value)

        gap_summary = GapSummary(
            covered=covered_count,
            total=len(results),
            coverage_percent=round(covered_count / len(results) * 100),
            covered_types=covered_types,
            missing_types=missing_types,
        )

    return FanoutResponse(
        target_query=request.target_query,
        model_used=model_name,
        total_sub_queries=len(results),
        sub_queries=results,
        gap_summary=gap_summary,
    )
