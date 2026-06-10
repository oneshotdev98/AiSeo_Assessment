"""POST /api/aeo/analyze — run the AEO check pipeline over a URL or pasted content.

The endpoint is a sync `def`: every step (httpx fetch aside) is CPU-bound
spaCy/textstat work, so FastAPI runs it on the threadpool and the event loop
stays free for concurrent requests.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import AEOAnalyzeRequest, AEOAnalyzeResponse, InputType
from app.services.aeo_checks import (
    BaseCheck,
    DirectAnswerCheck,
    HtagHierarchyCheck,
    ReadabilityCheck,
)
from app.services.content_parser import fetch_url, parse_content

router = APIRouter()

# Registry of checks, run in order. Adding a check = adding a class here.
CHECKS: list[BaseCheck] = [
    DirectAnswerCheck(),
    HtagHierarchyCheck(),
    ReadabilityCheck(),
]

SCORE_BANDS = [
    (85, "AEO Optimized"),
    (65, "Needs Improvement"),
    (40, "Significant Gaps"),
    (0, "Not AEO Ready"),
]


def band_for_score(score: int) -> str:
    for floor, label in SCORE_BANDS:
        if score >= floor:
            return label
    return SCORE_BANDS[-1][1]


@router.post("/analyze", response_model=AEOAnalyzeResponse)
def analyze(request: AEOAnalyzeRequest) -> AEOAnalyzeResponse:
    raw = (
        fetch_url(request.input_value)
        if request.input_type is InputType.url
        else request.input_value
    )
    content = parse_content(raw)

    results = [check.run(content) for check in CHECKS]

    raw_score = sum(result.score for result in results)
    max_total = sum(check.max_score for check in CHECKS)
    aeo_score = round(raw_score / max_total * 100)

    return AEOAnalyzeResponse(
        aeo_score=aeo_score,
        band=band_for_score(aeo_score),
        checks=results,
    )
