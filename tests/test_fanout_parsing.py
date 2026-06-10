"""Tests for the fan-out engine's LLM response parsing, validation and retry
logic — with the LLM fully mocked. No network calls are made."""

import json

import pytest

from app.models.schemas import SubQueryType
from app.services.fanout_engine import (
    LLMUnavailableError,
    build_prompt,
    critique,
    extract_json_object,
    generate_sub_queries,
    parse_llm_response,
)

pytestmark = pytest.mark.asyncio


def valid_payload(per_type: int = 2) -> dict:
    return {
        "sub_queries": [
            {"type": t.value, "query": f"sample {t.value} query number {i} about CRM software"}
            for t in SubQueryType
            for i in range(per_type)
        ]
    }


class FakeLLMClient:
    """Returns canned responses in order; records the prompts it received."""

    model_name = "fake-model"

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of canned responses")
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# JSON extraction & parsing
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_clean_json_passes_through(self):
        raw = json.dumps(valid_payload())
        assert json.loads(extract_json_object(raw)) == valid_payload()

    def test_markdown_fences_are_stripped(self):
        raw = "```json\n" + json.dumps(valid_payload()) + "\n```"
        assert json.loads(extract_json_object(raw)) == valid_payload()

    def test_surrounding_prose_is_sliced_away(self):
        raw = "Sure! Here is the JSON you asked for:\n" + json.dumps(valid_payload()) + "\nHope that helps!"
        assert json.loads(extract_json_object(raw)) == valid_payload()

    def test_no_json_object_raises(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json_object("I'm sorry, I can't produce JSON right now.")


class TestParsing:
    def test_valid_response_parses_all_queries(self):
        parsed = parse_llm_response(json.dumps(valid_payload()))
        assert len(parsed.sub_queries) == 12
        assert parsed.dropped == 0
        assert critique(parsed) is None

    def test_hallucinated_extra_fields_are_ignored(self):
        payload = valid_payload()
        payload["sub_queries"][0]["confidence"] = 0.99
        payload["sub_queries"][1]["reasoning"] = "because..."
        parsed = parse_llm_response(json.dumps(payload))

        assert len(parsed.sub_queries) == 12
        assert not hasattr(parsed.sub_queries[0], "confidence")

    def test_invalid_type_value_drops_only_that_item(self):
        payload = valid_payload()
        payload["sub_queries"].append({"type": "invented_type", "query": "a query with a fake type"})
        parsed = parse_llm_response(json.dumps(payload))

        assert len(parsed.sub_queries) == 12
        assert parsed.dropped == 1

    def test_duplicate_queries_are_deduplicated(self):
        payload = valid_payload()
        payload["sub_queries"].append(dict(payload["sub_queries"][0]))
        parsed = parse_llm_response(json.dumps(payload))

        assert len(parsed.sub_queries) == 12
        assert parsed.dropped == 1

    def test_missing_sub_queries_key_raises(self):
        with pytest.raises(ValueError):
            parse_llm_response(json.dumps({"queries": []}))

    def test_too_few_queries_is_flagged_by_critique(self):
        payload = {"sub_queries": valid_payload()["sub_queries"][:4]}
        parsed = parse_llm_response(json.dumps(payload))
        note = critique(parsed)

        assert note is not None
        assert "at least 10" in note


# ---------------------------------------------------------------------------
# Retry orchestration (mocked LLM)
# ---------------------------------------------------------------------------

class TestRetries:
    async def test_happy_path_returns_on_first_attempt(self):
        client = FakeLLMClient([json.dumps(valid_payload())])
        sub_queries, model = await generate_sub_queries("best CRM software", client=client)

        assert len(sub_queries) == 12
        assert model == "fake-model"
        assert len(client.prompts) == 1

    async def test_invalid_json_then_valid_json_recovers(self, monkeypatch):
        monkeypatch.setattr("app.services.fanout_engine.BACKOFF_BASE_SECONDS", 0)
        client = FakeLLMClient(["not json at all", json.dumps(valid_payload())])
        sub_queries, _ = await generate_sub_queries("best CRM software", client=client)

        assert len(sub_queries) == 12
        # The retry prompt must tell the model what went wrong.
        assert "rejected" in client.prompts[1]

    async def test_persistent_garbage_raises_llm_unavailable(self, monkeypatch):
        monkeypatch.setattr("app.services.fanout_engine.BACKOFF_BASE_SECONDS", 0)
        client = FakeLLMClient(["garbage", "garbage", "garbage"])

        with pytest.raises(LLMUnavailableError):
            await generate_sub_queries("best CRM software", client=client)

    async def test_short_but_usable_result_degrades_gracefully(self, monkeypatch):
        monkeypatch.setattr("app.services.fanout_engine.BACKOFF_BASE_SECONDS", 0)
        # 8 valid queries every time: below the 10 target, above the 6 floor.
        short = {"sub_queries": valid_payload()["sub_queries"][:8]}
        client = FakeLLMClient([json.dumps(short)] * 3)

        sub_queries, _ = await generate_sub_queries("best CRM software", client=client)
        assert len(sub_queries) == 8

    async def test_provider_exception_is_retried(self, monkeypatch):
        monkeypatch.setattr("app.services.fanout_engine.BACKOFF_BASE_SECONDS", 0)

        class FlakyClient(FakeLLMClient):
            calls = 0

            async def generate(self, prompt: str) -> str:
                FlakyClient.calls += 1
                if FlakyClient.calls == 1:
                    raise TimeoutError("simulated provider timeout")
                return await super().generate(prompt)

        client = FlakyClient([json.dumps(valid_payload())])
        sub_queries, _ = await generate_sub_queries("best CRM software", client=client)
        assert len(sub_queries) == 12


class TestPrompt:
    def test_prompt_contains_all_six_types_and_example(self):
        prompt = build_prompt("best CRM software")
        for t in SubQueryType:
            assert f'"{t.value}"' in prompt
        assert "best CRM software" in prompt
        assert '"sub_queries"' in prompt  # embedded example schema
