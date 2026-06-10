# AEGIS — AI Engineer Assignment Submission

A FastAPI service implementing two features from the AEGIS platform:

- **Feature 1 — AEO Content Scorer** (`POST /api/aeo/analyze`): three modular NLP checks (direct answer, H-tag hierarchy, readability) aggregated into a 0–100 AEO Readiness Score.
- **Feature 2 — Query Fan-Out Engine** (`POST /api/fanout/generate`): LLM-generated sub-queries across 6 types, with optional embedding-based content gap analysis.

The prompt iteration history is in [PROMPT_LOG.md](PROMPT_LOG.md).

---

## Running locally

```bash
# 1. Create a virtualenv (Python 3.11+)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the spaCy model
python -m spacy download en_core_web_sm

# 4. Configure secrets
cp .env.example .env
# edit .env and set GEMINI_API_KEY (or OPENAI_API_KEY)

# 5. Run the server
uvicorn app.main:app --reload
```

Interactive docs at http://127.0.0.1:8000/docs. Smoke tests:

```bash
curl -X POST http://127.0.0.1:8000/api/aeo/analyze \
  -H 'Content-Type: application/json' \
  -d '{"input_type": "text", "input_value": "<h1>Title</h1><p>A direct answer in one short sentence.</p>"}'

curl -X POST http://127.0.0.1:8000/api/fanout/generate \
  -H 'Content-Type: application/json' \
  -d '{"target_query": "best CRM software for small business"}'
```

### Tests

```bash
pytest        # 59 tests, all offline — no network or API key needed
```

The first `/api/fanout/generate` call with `existing_content` downloads the sentence-transformers model (~90 MB) — subsequent calls are fast.

### Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | one of these two | — | Gemini provider for fan-out |
| `OPENAI_API_KEY` | one of these two | — | OpenAI provider for fan-out |
| `LLM_PROVIDER` | no | inferred from keys | `gemini` or `openai` |
| `LLM_MODEL` | no | `gemini-2.5-flash` / `gpt-4o-mini` | model override |
| `EMBEDDING_MODEL` | no | `all-MiniLM-L6-v2` | sentence-transformers model |
| `SIMILARITY_THRESHOLD` | no | `0.72` | gap-analysis coverage cutoff |

Feature 1 needs no API key at all.

---

## Completed

**Completed:** everything in the brief — both endpoints, all three AEO checks with the exact scoring tables, score bands, LLM fan-out with retries/backoff/corrective re-prompting, per-item Pydantic validation of LLM output, semantic gap analysis with normalized embeddings, the 422/503 error envelopes, 59 offline tests (including the mocked-LLM bonus tests), and PROMPT_LOG.md.

**Deviations from the assignment brief (documented per the FAQ):**

- **Model:** the brief names `gemini-1.5-flash`, which Google has since retired (the API returns 404 for it). The default is `gemini-2.5-flash`; `model_used` in the response reflects whatever actually ran.
- **Graceful degradation on under-delivery:** if after 3 attempts the LLM never reaches 10 sub-queries but the best attempt produced ≥ 6 valid ones, the API returns that best set rather than a 503. A partial fan-out is more useful to a caller than an error; below 6 it is a real failure and the 503 envelope is returned.
- **Extra `details` fields:** `hedge_phrases_found` (which phrases matched, not just a boolean) and a `warning` on readability when the text is too short for FK to be reliable. Additive only — every field in the contract is present.

**Known limitations (honest list):**

- `en_core_web_sm` occasionally mis-tags noun/verb-ambiguous openings (e.g. "scores" parsed as a noun), which can mark a genuinely declarative sentence as non-declarative. `en_core_web_lg` would reduce this at the cost of a ~750 MB image; for a take-home, `sm` keeps setup fast (noted inline in the tests where it matters).
- Plain-text input has no headings, so Check B scores 0 for it. That is arguably correct (no extractable heading structure for an answer engine) but penalizes pasted prose; a production version might exclude Check B from the denominator for plain text.
- URL fetching does not render JavaScript. SPA pages that ship an empty body return a clean 422 (`content_unparseable`) rather than a misleading score.

---

## Key engineering decisions

### 1. LLM JSON reliability

Defense in depth, in order:

1. **Provider JSON mode** (`response_mime_type: application/json` on Gemini, `response_format: json_object` on OpenAI) so the decoder constrains output before the prompt even matters.
2. **Prompt-level schema pinning** — exact output shape with a full worked example (see PROMPT_LOG.md).
3. **Tolerant extraction** — markdown fences stripped, outermost `{...}` sliced out of any surrounding prose.
4. **Per-item Pydantic validation** (`LLMSubQuery`, `extra="ignore"`, enum-typed `type`): hallucinated extra fields are dropped silently, an invented query type rejects only that item — one bad element never fails the batch. Duplicates are removed.
5. **Retry with corrective feedback**: up to 3 attempts with exponential backoff; each rejected attempt re-prompts with a specific description of what was wrong ("only 8 valid sub-queries; at least 10 required"). This converges far faster than blind resampling.
6. **Graceful floor**: best attempt with ≥ 6 valid queries is returned; otherwise a structured 503 — the API never crashes on bad model output (tested with pure garbage responses).

### 2. Embedding model choice

`all-MiniLM-L6-v2`, swappable via env var. It is ~5× faster than `all-mpnet-base-v2`, and this endpoint embeds every sentence of an article synchronously inside a request — latency dominates. MiniLM trails mpnet by a few points on STS benchmarks, but the task here is a coarse covered/not-covered decision against a threshold, not fine-grained ranking, so the accuracy gap rarely flips an outcome. In production I'd pre-compute and cache content embeddings (they change rarely), at which point upgrading to mpnet — or an API embedding model — costs nothing per request.

Cosine similarity is computed correctly: vectors are encoded with `normalize_embeddings=True`, so the dot product *is* the cosine similarity (exactly, not approximately).

### 3. Similarity threshold (0.72)

Kept as the default but configurable (`SIMILARITY_THRESHOLD`). Spot-checking with MiniLM suggests it is reasonable but slightly strict: genuinely on-topic sentences typically score 0.72–0.88 against a matching sub-query, while topically-adjacent-but-not-answering content lands 0.55–0.70. The right way to tune it is empirical, not aesthetic: build a small labeled set (sub-query, content, human covered/not-covered judgment), sweep the threshold, and pick the value that maximizes F1 — also checking whether the optimum differs per query type (`definitional` queries match more literally than `use_case` ones, so per-type thresholds may beat a global one). The threshold is also model-dependent — swapping to mpnet shifts the score distribution, so it must be re-tuned with any model change.

### 4. Content parsing robustness

- **No clear first paragraph:** first `<p>` with ≥ 3 words after boilerplate stripping; if none exists, fall back to the first text block; if the page yields nothing, Check A scores 0 with an actionable recommendation rather than erroring.
- **JS-heavy / empty pages:** if fewer than 5 words of text survive parsing, the API returns 422 `content_unparseable` with a message naming the likely causes (JS rendering, login wall) instead of scoring garbage.
- **Login walls / errors:** non-2xx responses, timeouts (10 s), non-HTML content types, and oversized pages (> 2 MB) all map to 422 `url_fetch_failed` with detail.
- **Boilerplate:** nav/footer/header/aside/script/etc. removed before paragraph extraction and readability scoring — but headings are collected *before* stripping, because an `<h1>` inside a `<header>` wrapper is still the page's H1.
- **Plain text:** detected heuristically (no structural tags) and split on blank lines.

### 5. Failure modes

- LLM timeout mid-request: each provider call has a 30 s timeout; a timeout counts as a failed attempt and is retried with backoff. After 3 failures: structured 503, never an unhandled 500.
- All custom exceptions (`ContentFetchError`, `ContentParseError`, `LLMUnavailableError`) are mapped to the error envelope (`error`/`message`/`detail`) by FastAPI exception handlers in `app/main.py`.

### 6. Concurrency model

Mixed deliberately, per workload:

- `/api/aeo/analyze` is a **sync `def`** route: spaCy and textstat are CPU-bound, so FastAPI runs it on the threadpool and the event loop stays free.
- `/api/fanout/generate` is **async**: the LLM call is pure I/O and awaits cleanly; the CPU-bound embedding step is explicitly pushed to the threadpool via `run_in_threadpool` so it can't block the loop.

### 7. spaCy usage (why, not just `nlp(text)`)

Check A needs to distinguish a *complete declarative sentence* from a question, fragment, or imperative. Surface heuristics catch questions ("ends with ?") but not fragments like "The best AI writing tools of 2025." The dependency parse answers this directly: a declarative clause has a verbal ROOT (`VERB`/`AUX`) governing an explicit subject arc (`nsubj`/`nsubjpass`/`expl`/`csubj`). A noun-phrase fragment has a noun ROOT; an imperative has a verb ROOT with no subject. spaCy's sentencizer also drives sentence chunking for gap analysis and complex-sentence ranking.

---

## Prompt design (summary — full history in PROMPT_LOG.md)

The final prompt: a system message fixing the role and the single-JSON-object output rule; a user message with the count window (10–15), the 6 types as exact snake_case literals with one-line definitions, 5 numbered hard requirements (≥ 2 per type, exactly two keys per object, never invent types), and a full worked example for a **deliberately unrelated topic** (project management software) so the model imitates the format without leaking example content into its output. What a prompt cannot guarantee (counts, valid JSON every time) is enforced outside the prompt — validation, corrective retries, and a degradation floor.

---

## What I'd improve with more time

1. **Strict structured outputs** (OpenAI `strict: true` / Gemini `response_schema`) to machine-enforce the schema and delete most of the extraction code.
2. **Threshold calibration**: the labeled-set + F1 sweep described above, with per-type thresholds.
3. **Embedding cache** keyed by content hash so repeat analyses of the same article skip re-encoding.
4. **Paragraph-window chunking** (2–3 sentence sliding windows) alongside sentence chunks — some sub-queries are answered across sentence boundaries that single-sentence chunks miss.
5. **`en_core_web_lg`** (or a transformer pipeline) behind the existing `get_nlp()` seam to fix noun/verb-ambiguity misparses in Check A.
6. **Trafilatura** for boilerplate extraction — the tag-based stripping here handles semantic HTML well but loses to readability-style algorithms on messy real-world pages.
