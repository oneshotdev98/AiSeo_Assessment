# Prompt Iteration Log — Query Fan-Out Engine

How the fan-out prompt evolved from first draft to the version shipped in
[app/services/fanout_engine.py](app/services/fanout_engine.py). Each revision
targets a specific, named failure mode.

---

## Draft 1 — the naive version

```
Generate 10-15 sub-queries for the search query "{target_query}".
Use these types: comparative, feature_specific, use_case, trust_signals,
how_to, definitional. Return JSON.
```

**What was wrong with it:**

1. **Markdown fences.** "Return JSON" reliably produces ` ```json ... ``` `
   blocks and often a "Here is the JSON you asked for:" preamble —
   `json.loads` fails on the raw response.
2. **Unpinned schema.** Without an explicit shape, the model invents its own:
   sometimes a top-level array, sometimes `{"queries": [...]}`, sometimes
   objects with extra keys (`"relevance"`, `"explanation"`) it decided would
   be helpful.
3. **Type distribution ignored.** Nothing requires 2 per type, so the model
   front-loads the easy types (`how_to`, `definitional`) and returns zero
   `trust_signals` queries.
4. **Type drift.** With types listed only as prose, the model occasionally
   "improves" them — returning `"how-to"` (hyphen), `"comparison"`, or
   inventing `"pricing"` as a seventh type.

---

## Draft 2 — added schema and counts

Changes: specified the exact output shape `{"sub_queries": [{"type", "query"}]}`,
required "at least 2 of each of the 6 types", listed allowed type values in
quotes, and added "no markdown, JSON only".

**Better, but still broken in two ways:**

1. **Example anchoring.** I initially included an example for the *same*
   topic as a likely user query (AI writing tools). The model then leaked
   example phrasing into outputs for unrelated topics — a CRM query came back
   with sub-queries about "content optimization". Fix: the embedded example
   uses a deliberately **different topic** (project management software) so
   the model imitates the *format*, never the content.
2. **Under-delivery.** Instruction-only count requirements ("at least 10")
   are followed maybe 80–90% of the time; the model sometimes stops at 8.
   A prompt alone can't fix a stochastic failure — this needs to be handled
   *outside* the prompt.

---

## Draft 3 (final) — structure + an out-of-band correction loop

Final prompt structure (full text in `fanout_engine.py`):

1. **System message** establishing the role (query fan-out engine inside an
   AEO platform, simulating AI search decomposition) and the hard output
   rule: *a single valid JSON object, nothing else*.
2. **User message** with: the target query; the count window (10–15); the 6
   types as exact snake_case string literals with one-line definitions;
   5 numbered hard requirements (≥2 per type, exactly two keys per object,
   never invent types, no markdown); and a **full worked example for an
   unrelated topic** showing 12 correctly-distributed sub-queries.
3. **Provider JSON mode** (`response_mime_type` / `response_format`) so the
   decoder itself constrains output to JSON — the prompt is no longer the
   only line of defense.

And critically, the parts a prompt cannot guarantee moved into code:

- **Per-item Pydantic validation** with an enum on `type` — hallucinated
  fields are stripped, invented types drop only that item.
- **Retry with corrective feedback**: a rejected attempt re-prompts with
  *"your previous response was rejected for this reason: only 8 valid
  sub-queries were returned; at least 10 are required..."*. Telling the model
  what specifically failed converges in 1 retry far more often than blind
  resampling.
- **Graceful degradation** below target but above a usable floor (≥6), and a
  structured 503 below that.

**Result:** with JSON mode + this prompt, parse failures are rare; when count
or distribution slips, the correction loop recovers it. The test suite
([tests/test_fanout_parsing.py](tests/test_fanout_parsing.py)) locks in every
failure mode above with a mocked LLM: fenced output, prose-wrapped JSON,
invented types, extra fields, duplicates, under-delivery, and full garbage.

## What I'd try next

- **Strict structured outputs** (OpenAI `strict: true` JSON schema / Gemini
  `response_schema`) — would make the schema machine-enforced and let me
  delete most of the extraction code. Kept prompt-level enforcement here to
  stay provider-portable per the assignment ("you may use any LLM").
- **Few-shot with a negative example** ("do NOT return: ...") — I avoided it
  for now; negative examples sometimes *teach* the failure pattern.
- **Temperature sweep**: 0.4 balances phrasing diversity across the 6 types
  against format discipline; 0 made sub-queries within a type near-duplicates.
