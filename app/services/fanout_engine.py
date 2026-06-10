"""Query Fan-Out Engine — LLM prompt, provider clients, defensive parsing.

Design notes
------------
- The prompt pins the exact JSON schema, enumerates the 6 allowed types,
  requires >= 2 sub-queries per type (12–15 total), and embeds a concrete
  example for a DIFFERENT topic than the user's (so the model imitates the
  format, not the content).
- Both providers are called in JSON mode (response_mime_type /
  response_format) which removes most markdown-fence failures, but we still
  defensively strip fences and slice to the outermost braces because JSON
  mode is not a guarantee.
- Every sub-query is validated individually with Pydantic (LLMSubQuery).
  Invalid items are dropped rather than failing the batch; hallucinated
  extra fields are ignored; unknown `type` values are rejected by the enum.
- Retries: up to 3 attempts with exponential backoff. A failed attempt
  appends corrective feedback to the next prompt describing exactly what was
  wrong (bad JSON / too few queries / missing types). If all attempts fall
  short but the best attempt produced a usable set (>= MIN_ACCEPTABLE),
  we degrade gracefully and return it; otherwise we raise LLMUnavailableError
  which the API maps to a 503.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.models.schemas import LLMSubQuery, SubQueryType

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.8
REQUEST_TIMEOUT_SECONDS = 30

TARGET_MIN_QUERIES = 10
TARGET_MAX_QUERIES = 15
MIN_PER_TYPE = 2
# If after all retries we still have fewer than TARGET_MIN_QUERIES but at
# least this many valid ones, return them instead of failing the request.
MIN_ACCEPTABLE_QUERIES = 6

QUERY_TYPES = [t.value for t in SubQueryType]


class LLMUnavailableError(Exception):
    """The LLM could not produce a usable response after retries."""

    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a query fan-out engine inside an Answer Engine Optimization platform.
Given a target search query, you simulate how an AI search engine (like
Perplexity or Google AI Mode) decomposes it into the sub-queries it would
research to build a comprehensive answer.

You must respond with a single valid JSON object and NOTHING else:
no markdown code fences, no commentary, no trailing text.
"""

USER_PROMPT_TEMPLATE = """\
TARGET QUERY: "{target_query}"

Generate between {min_total} and {max_total} sub-queries that an AI search \
engine would research to answer the target query comprehensively.

## Allowed sub-query types (use EXACTLY these snake_case values)
- "comparative": the subject vs. alternatives or substitutes
- "feature_specific": a specific capability, attribute, or specification
- "use_case": a concrete real-world application or audience
- "trust_signals": reviews, case studies, credibility, social proof
- "how_to": procedural or instructional phrasing
- "definitional": conceptual, "what is" style phrasing

## Hard requirements
1. Include AT LEAST {min_per_type} sub-queries of EACH of the 6 types.
2. Every sub-query must be specific to the target query's topic and phrased
   the way a real user or AI agent would search it.
3. Each object has EXACTLY two keys: "type" and "query". No other keys.
4. "type" must be one of the 6 values listed above — never invent new types.
5. Output one JSON object with a single key "sub_queries". No markdown.

## Example output (for the unrelated target query "best project management software")
{{
  "sub_queries": [
    {{"type": "comparative", "query": "Asana vs Monday vs ClickUp for small teams"}},
    {{"type": "comparative", "query": "project management software vs spreadsheets for tracking work"}},
    {{"type": "feature_specific", "query": "project management tool with Gantt charts and time tracking"}},
    {{"type": "feature_specific", "query": "project management software with API integrations"}},
    {{"type": "use_case", "query": "project management software for remote engineering teams"}},
    {{"type": "use_case", "query": "managing client projects in an agency with PM software"}},
    {{"type": "trust_signals", "query": "Asana customer reviews and case studies 2025"}},
    {{"type": "trust_signals", "query": "most trusted project management tools according to G2"}},
    {{"type": "how_to", "query": "how to migrate a team from spreadsheets to project management software"}},
    {{"type": "how_to", "query": "how to set up sprint planning in a PM tool"}},
    {{"type": "definitional", "query": "what is project management software"}},
    {{"type": "definitional", "query": "what does work breakdown structure mean in project management"}}
  ]
}}

Now produce the JSON object for the target query: "{target_query}"\
"""


def build_prompt(target_query: str, corrective_note: str | None = None) -> str:
    prompt = USER_PROMPT_TEMPLATE.format(
        target_query=target_query.strip(),
        min_total=TARGET_MIN_QUERIES,
        max_total=TARGET_MAX_QUERIES,
        min_per_type=MIN_PER_TYPE,
    )
    if corrective_note:
        prompt += (
            "\n\nIMPORTANT — your previous response was rejected for this reason: "
            f"{corrective_note} Fix this and respond again with ONLY the JSON object."
        )
    return prompt


# ---------------------------------------------------------------------------
# Response parsing & validation
# ---------------------------------------------------------------------------

@dataclass
class ParsedFanout:
    sub_queries: list[LLMSubQuery]
    dropped: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def types_present(self) -> set[SubQueryType]:
        return {sq.type for sq in self.sub_queries}

    def missing_type_counts(self) -> dict[str, int]:
        """Types that have fewer than MIN_PER_TYPE sub-queries."""
        counts = {t: 0 for t in SubQueryType}
        for sq in self.sub_queries:
            counts[sq.type] += 1
        return {
            t.value: c for t, c in counts.items() if c < MIN_PER_TYPE
        }


def extract_json_object(raw: str) -> str:
    """Best-effort extraction of a JSON object from LLM output.

    Handles markdown fences and stray prose before/after the object by
    slicing from the first '{' to the last '}'.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found in response", text, 0)
    return text[start : end + 1]


def parse_llm_response(raw: str) -> ParsedFanout:
    """Parse + validate the LLM response. Raises json.JSONDecodeError or ValueError."""
    payload = json.loads(extract_json_object(raw))

    if not isinstance(payload, dict) or not isinstance(payload.get("sub_queries"), list):
        raise ValueError('Response JSON must be an object with a "sub_queries" list.')

    valid: list[LLMSubQuery] = []
    dropped = 0
    seen_queries: set[str] = set()
    for item in payload["sub_queries"]:
        try:
            sub_query = LLMSubQuery.model_validate(item)
        except ValidationError as exc:
            dropped += 1
            logger.warning("Dropping invalid sub-query %r: %s", item, exc)
            continue
        # de-duplicate (LLMs repeat themselves under pressure to hit counts)
        key = sub_query.query.lower()
        if key in seen_queries:
            dropped += 1
            continue
        seen_queries.add(key)
        valid.append(sub_query)

    if not valid:
        raise ValueError("No valid sub-queries in response.")

    return ParsedFanout(sub_queries=valid, dropped=dropped)


def critique(parsed: ParsedFanout) -> str | None:
    """Return a corrective note for the next attempt, or None if acceptable."""
    problems: list[str] = []
    if len(parsed.sub_queries) < TARGET_MIN_QUERIES:
        problems.append(
            f"only {len(parsed.sub_queries)} valid sub-queries were returned; "
            f"at least {TARGET_MIN_QUERIES} are required"
        )
    shortfall = parsed.missing_type_counts()
    if shortfall:
        problems.append(
            "these types had fewer than "
            f"{MIN_PER_TYPE} sub-queries: {', '.join(sorted(shortfall))}"
        )
    if parsed.dropped:
        problems.append(
            f"{parsed.dropped} items were rejected (wrong keys, invalid type "
            "value, or duplicates)"
        )
    return "; ".join(problems) + "." if problems else None


# ---------------------------------------------------------------------------
# LLM provider clients
# ---------------------------------------------------------------------------

class GeminiClient:
    def __init__(self, model: str):
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name = model
        self._model = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.4,
            },
        )

    async def generate(self, prompt: str) -> str:
        response = await self._model.generate_content_async(
            prompt, request_options={"timeout": REQUEST_TIMEOUT_SECONDS}
        )
        return response.text


class OpenAIClient:
    def __init__(self, model: str):
        from openai import AsyncOpenAI

        self.model_name = model
        self._client = AsyncOpenAI(timeout=REQUEST_TIMEOUT_SECONDS)

    async def generate(self, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        return response.choices[0].message.content or ""


def get_llm_client():
    """Pick a provider from LLM_PROVIDER, falling back to whichever key is set."""
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if not provider:
        if os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise LLMUnavailableError(
                "No LLM provider configured.",
                detail="Set GEMINI_API_KEY or OPENAI_API_KEY (and optionally LLM_PROVIDER/LLM_MODEL).",
            )
    if provider == "gemini":
        return GeminiClient(model=os.getenv("LLM_MODEL", "gemini-2.5-flash"))
    if provider == "openai":
        return OpenAIClient(model=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    raise LLMUnavailableError(
        f"Unknown LLM_PROVIDER '{provider}'.", detail="Use 'gemini' or 'openai'."
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def generate_sub_queries(
    target_query: str, client=None
) -> tuple[list[LLMSubQuery], str]:
    """Generate validated sub-queries. Returns (sub_queries, model_name).

    `client` is injectable for tests; anything with an async generate(prompt)
    method and a model_name attribute works.
    """
    if client is None:
        client = get_llm_client()

    best: ParsedFanout | None = None
    corrective_note: str | None = None
    last_error = "unknown"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            await asyncio.sleep(BACKOFF_BASE_SECONDS * 2 ** (attempt - 2))
        try:
            raw = await client.generate(build_prompt(target_query, corrective_note))
        except Exception as exc:  # provider/network error — retry
            last_error = f"{type(exc).__name__} on attempt {attempt}: {exc}"
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, exc)
            corrective_note = None
            continue

        try:
            parsed = parse_llm_response(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"{type(exc).__name__} on attempt {attempt}"
            logger.warning("Unparseable LLM response (attempt %d): %s", attempt, exc)
            corrective_note = (
                "the response was not a single valid JSON object with a "
                '"sub_queries" list'
            )
            continue

        if best is None or len(parsed.sub_queries) > len(best.sub_queries):
            best = parsed

        corrective_note = critique(parsed)
        if corrective_note is None:
            return parsed.sub_queries, client.model_name
        last_error = f"incomplete result on attempt {attempt}: {corrective_note}"

    # Out of attempts: degrade gracefully if the best attempt is usable.
    if best is not None and len(best.sub_queries) >= MIN_ACCEPTABLE_QUERIES:
        logger.warning(
            "Returning best-effort fan-out with %d sub-queries after %d attempts",
            len(best.sub_queries), MAX_ATTEMPTS,
        )
        return best.sub_queries, client.model_name

    raise LLMUnavailableError(
        f"Fan-out generation failed. The LLM returned an invalid response "
        f"after {MAX_ATTEMPTS} retries.",
        detail=last_error,
    )
