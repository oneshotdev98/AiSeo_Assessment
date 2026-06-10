"""Unit tests for Check C — Snippet Readability."""

import pytest

from app.services.aeo_checks.readability import (
    ReadabilityCheck,
    rank_complex_sentences,
    score_for_grade,
)
from app.services.content_parser import ParsedContent

# Engineered to land in the FK 7-9 band (measures ~7.5): 15-18 word
# sentences, mostly one/two-syllable words with a few longer ones.
GRADE_7_9_TEXT = """
AI search engines now answer questions directly instead of showing a list of links.
This change means your content must give a clear answer near the top of the page.
Teams that structure their writing for these engines receive more citations and more visits over time.
A successful page begins with a concise answer, then explains the details in plain language below it.
Every section needs a clear heading so the engine can locate and extract the right passage.
"""

# Dense academic prose: long sentences, polysyllabic vocabulary -> FK >= 12.
GRADE_12_PLUS_TEXT = """
The multifaceted nature of contemporary content optimization necessitates a comprehensive
reconceptualization of established methodological paradigms, particularly insofar as
algorithmic intermediaries increasingly disintermediate traditional informational
retrieval architectures. Heuristically speaking, syntactic structures characterized by
excessive subordination and nominalization demonstrably attenuate extractability,
notwithstanding their perceived epistemological sophistication within academically
oriented discursive communities.
"""


def make_content(text: str) -> ParsedContent:
    return ParsedContent(first_paragraph=text.strip().split("\n")[0], main_text=text.strip())


@pytest.fixture
def check() -> ReadabilityCheck:
    return ReadabilityCheck()


class TestPassingCase:
    def test_grade_7_to_9_text_scores_full_marks(self, check):
        result = check.run(make_content(GRADE_7_9_TEXT))

        assert 7 <= result.details["fk_grade_level"] <= 9
        assert result.score == 20
        assert result.passed is True
        assert result.recommendation is None
        assert result.details["target_range"] == "7-9"


class TestFailingCase:
    def test_dense_academic_text_scores_0(self, check):
        result = check.run(make_content(GRADE_12_PLUS_TEXT))

        assert result.details["fk_grade_level"] >= 12
        assert result.score == 0
        assert result.passed is False
        assert "Grade" in result.recommendation

    def test_complex_sentences_are_reported(self, check):
        result = check.run(make_content(GRADE_12_PLUS_TEXT))

        sentences = result.details["complex_sentences"]
        assert 1 <= len(sentences) <= 3
        assert all(isinstance(s, str) and s for s in sentences)


class TestScoringBands:
    @pytest.mark.parametrize(
        ("grade", "expected"),
        [
            (7.0, 20), (8.4, 20), (9.0, 20),     # target band
            (6.0, 14), (6.9, 14), (9.5, 14), (10.0, 14),  # one band out
            (5.0, 8), (5.9, 8), (10.1, 8), (11.0, 8),     # two bands out
            (4.0, 0), (3.2, 0), (11.1, 0), (12.0, 0), (15.7, 0),
        ],
    )
    def test_score_for_grade(self, grade, expected):
        assert score_for_grade(grade) == expected


class TestComplexSentenceRanking:
    def test_ranks_by_syllable_density(self):
        text = (
            "The cat sat on the mat near the door.\n"
            "Organizational prioritization methodologies facilitate institutional decision frameworks.\n"
            "Dogs run fast in the park every day."
        )
        top = rank_complex_sentences(text, top_n=1)
        assert "Organizational" in top[0]

    def test_short_fragments_are_ignored(self):
        text = "Pricing.\nFeatures.\nThe platform helps teams write content that AI engines can cite."
        top = rank_complex_sentences(text)
        assert len(top) == 1
