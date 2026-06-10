"""AEGIS — AI Engineer Assignment API."""

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()  # pick up GEMINI_API_KEY etc. from a local .env file

from app.api import aeo, fanout
from app.models.schemas import APIError
from app.services.content_parser import ContentFetchError, ContentParseError
from app.services.fanout_engine import LLMUnavailableError

app = FastAPI(
    title="AEGIS — Answer Engine & Generative Intelligence Suite",
    description="AEO content scoring and LLM query fan-out with semantic gap analysis.",
    version="0.1.0",
)

app.include_router(aeo.router, prefix="/api/aeo", tags=["aeo"])
app.include_router(fanout.router, prefix="/api/fanout", tags=["fanout"])


def _error_response(status_code: int, error: str, message: str, detail: str | None) -> JSONResponse:
    body = APIError(error=error, message=message, detail=detail)
    return JSONResponse(status_code=status_code, content=body.model_dump())


@app.exception_handler(ContentFetchError)
async def handle_fetch_error(_request: Request, exc: ContentFetchError) -> JSONResponse:
    return _error_response(422, "url_fetch_failed", exc.message, exc.detail)


@app.exception_handler(ContentParseError)
async def handle_parse_error(_request: Request, exc: ContentParseError) -> JSONResponse:
    return _error_response(422, "content_unparseable", exc.message, exc.detail)


@app.exception_handler(LLMUnavailableError)
async def handle_llm_error(_request: Request, exc: LLMUnavailableError) -> JSONResponse:
    return _error_response(503, "llm_unavailable", exc.message, exc.detail)


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service": "AEGIS AI Engineer Assignment",
        "endpoints": ["POST /api/aeo/analyze", "POST /api/fanout/generate"],
        "docs": "/docs",
    }


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}
