"""Check C — Snippet Readability.

Scores Flesch-Kincaid Grade Level against the 7–9 sweet spot for AI answer
extraction, and surfaces the 3 most complex sentences (ranked by mean
syllables per word) so the writer knows exactly what to simplify.

The score bands in the spec name integer grades ("6 or 10" -> 14). FK grade
is continuous, so we band the continuous value symmetrically around the
target range: [6,7) or (9,10] -> 14, [5,6) or (10,11] -> 8, beyond -> 0.
"""

from __future__ import annotations

import textstat

from app.models.schemas import CheckResult
from app.services.aeo_checks.base import BaseCheck
from app.services.content_parser import ParsedContent
from app.services.nlp import get_nlp

TARGET_RANGE = "7-9"
MIN_WORDS_FOR_RELIABLE_SCORE = 30


def score_for_grade(grade: float) -> int:
    if 7 <= grade <= 9:
        return 20
    if 6 <= grade < 7 or 9 < grade <= 10:
        return 14
    if 5 <= grade < 6 or 10 < grade <= 11:
        return 8
    return 0


def rank_complex_sentences(text: str, top_n: int = 3) -> list[str]:
    """Top-N sentences by syllable density (syllables ÷ words).

    Blocks are processed line by line: main_text joins paragraphs and
    headings with newlines, and running spaCy across those boundaries glues
    a heading onto the sentence that follows it.
    """
    nlp = get_nlp()
    scored: list[tuple[float, str]] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        for sent in nlp(block).sents:
            words = [t.text for t in sent if t.is_alpha]
            if len(words) < 4:  # headings/fragments aren't meaningful here
                continue
            syllables = sum(textstat.syllable_count(w) for w in words)
            scored.append((syllables / len(words), sent.text.strip()))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [sentence for _density, sentence in scored[:top_n]]


class ReadabilityCheck(BaseCheck):
    check_id = "readability"
    name = "Snippet Readability"

    def run(self, content: ParsedContent) -> CheckResult:
        # Boilerplate was already stripped by the content parser; replace the
        # block separators so textstat sees sentence boundaries.
        text = content.main_text.replace("\n", " ").strip()
        word_count = len(text.split())

        grade = round(textstat.flesch_kincaid_grade(text), 1)
        score = score_for_grade(grade)
        complex_sentences = rank_complex_sentences(content.main_text)

        details: dict = {
            "fk_grade_level": grade,
            "target_range": TARGET_RANGE,
            "complex_sentences": complex_sentences,
        }
        if word_count < MIN_WORDS_FOR_RELIABLE_SCORE:
            details["warning"] = (
                f"Only {word_count} words of content; FK grade is unreliable "
                "on very short texts."
            )

        recommendation = None
        if score < self.max_score:
            if grade > 9:
                recommendation = (
                    f"Content reads at Grade {grade}. Shorten sentences and "
                    "replace technical jargon with plain language to reach "
                    f"Grade {TARGET_RANGE}."
                )
            else:
                recommendation = (
                    f"Content reads at Grade {grade}, below the {TARGET_RANGE} "
                    "target. Slightly longer, more substantive sentences will "
                    "make it credible enough for AI engines to cite."
                )

        return self._result(score=score, details=details, recommendation=recommendation)
