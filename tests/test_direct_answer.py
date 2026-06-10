"""Unit tests for Check A — Direct Answer Detection.

The check is exercised directly (not through the endpoint) with hand-built
ParsedContent, so failures point at the check logic itself.
"""

import pytest

from app.services.aeo_checks.direct_answer import (
    DirectAnswerCheck,
    find_hedge_phrases,
    is_declarative,
)
from app.services.content_parser import ParsedContent


def make_content(first_paragraph: str) -> ParsedContent:
    return ParsedContent(first_paragraph=first_paragraph, main_text=first_paragraph)


@pytest.fixture
def check() -> DirectAnswerCheck:
    return DirectAnswerCheck()


class TestPassingCase:
    def test_short_declarative_answer_scores_full_marks(self, check):
        paragraph = (
            "AEO content scoring shows how easily AI search engines can "
            "extract a direct answer from your page. A strong page opens with "
            "a short, confident statement."
        )
        result = check.run(make_content(paragraph))

        assert result.score == 20
        assert result.passed is True
        assert result.details["word_count"] <= 60
        assert result.details["is_declarative"] is True
        assert result.details["has_hedge_phrase"] is False
        assert result.recommendation is None


class TestFailingCases:
    def test_hedge_phrase_caps_score_at_12(self, check):
        paragraph = (
            "The best tool depends on your goals, and generally speaking "
            "results may vary between teams."
        )
        result = check.run(make_content(paragraph))

        assert result.score == 12
        assert result.passed is False
        assert result.details["has_hedge_phrase"] is True
        assert "generally speaking" in result.details["hedge_phrases_found"]
        assert "may vary" in result.details["hedge_phrases_found"]
        assert result.recommendation is not None

    def test_question_opening_is_not_declarative(self, check):
        result = check.run(make_content("What is the best AI writing tool for SEO?"))

        assert result.details["is_declarative"] is False
        assert result.score == 12

    def test_61_to_90_words_scores_8(self, check):
        paragraph = " ".join(["Content optimization matters for modern teams."] * 11)  # 66 words
        result = check.run(make_content(paragraph))

        assert 61 <= result.details["word_count"] <= 90
        assert result.score == 8

    def test_over_90_words_scores_0(self, check):
        paragraph = " ".join(["Content optimization matters for modern teams."] * 16)  # 96 words
        result = check.run(make_content(paragraph))

        assert result.details["word_count"] > 90
        assert result.score == 0

    def test_missing_first_paragraph_scores_0_with_recommendation(self, check):
        result = check.run(make_content(""))

        assert result.score == 0
        assert result.recommendation is not None


class TestHelpers:
    def test_find_hedge_phrases_is_case_insensitive(self):
        assert find_hedge_phrases("Well, It Depends on the use case.") == ["it depends"]

    def test_declarative_sentence_detected(self):
        # Uses an unambiguous verb: en_core_web_sm misparses noun/verb-ambiguous
        # words like "scores" as nouns (documented limitation in the README).
        assert is_declarative("The platform analyzes content for AI readiness.") is True

    def test_noun_fragment_is_not_declarative(self):
        assert is_declarative("The best AI writing tools of 2025.") is False

    def test_imperative_is_not_declarative(self):
        assert is_declarative("Use shorter sentences in your opening paragraph.") is False
