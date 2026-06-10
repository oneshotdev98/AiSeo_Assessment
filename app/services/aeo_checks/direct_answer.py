"""Check A — Direct Answer Detection.

Does the first paragraph answer the likely primary query in <= 60 words,
as a clear declarative statement, without hedging?

spaCy usage (and why): we need to distinguish a *complete declarative
sentence* from a question or fragment. Surface heuristics (ends with "?")
catch questions but not fragments like "The best AI writing tools of 2025."
The dependency parse gives us this directly: a declarative sentence has a
root verb (ROOT with VERB/AUX pos) governing an explicit subject
(nsubj/nsubjpass/expl/csubj). A noun-phrase fragment has a noun ROOT and no
subject arc; an imperative has a verb ROOT but no subject.
"""

from __future__ import annotations

from app.models.schemas import CheckResult
from app.services.aeo_checks.base import BaseCheck
from app.services.content_parser import ParsedContent
from app.services.nlp import get_nlp

WORD_LIMIT = 60
SOFT_WORD_LIMIT = 90

HEDGE_PHRASES = (
    "it depends",
    "may vary",
    "in some cases",
    "this varies",
    "generally speaking",
)

SUBJECT_DEPS = {"nsubj", "nsubjpass", "expl", "csubj", "csubjpass"}


def find_hedge_phrases(text: str) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in HEDGE_PHRASES if phrase in lowered]


def is_declarative(text: str) -> bool:
    """True if the paragraph opens with a complete declarative statement.

    We judge the first sentence: that is what an answer engine extracts as
    the direct answer. Later sentences may elaborate freely.
    """
    doc = get_nlp()(text)
    sentences = list(doc.sents)
    if not sentences:
        return False

    first = sentences[0]
    if first.text.strip().endswith("?"):
        return False

    root = first.root
    # A declarative clause needs a verbal root...
    if root.pos_ not in ("VERB", "AUX"):
        return False
    # ...with an explicit subject (rules out imperatives and fragments).
    return any(child.dep_ in SUBJECT_DEPS for child in root.children)


class DirectAnswerCheck(BaseCheck):
    check_id = "direct_answer"
    name = "Direct Answer Detection"

    def run(self, content: ParsedContent) -> CheckResult:
        paragraph = content.first_paragraph.strip()

        if not paragraph:
            return self._result(
                score=0,
                details={
                    "word_count": 0,
                    "threshold": WORD_LIMIT,
                    "is_declarative": False,
                    "has_hedge_phrase": False,
                },
                recommendation=(
                    "No opening paragraph could be identified. Start the page "
                    "with a short paragraph that directly answers the primary query."
                ),
            )

        word_count = len(paragraph.split())
        hedges_found = find_hedge_phrases(paragraph)
        has_hedge = bool(hedges_found)
        declarative = is_declarative(paragraph)

        if word_count <= WORD_LIMIT:
            score = 20 if (declarative and not has_hedge) else 12
        elif word_count <= SOFT_WORD_LIMIT:
            score = 8
        else:
            score = 0

        recommendation = self._recommend(word_count, declarative, hedges_found)

        details = {
            "word_count": word_count,
            "threshold": WORD_LIMIT,
            "is_declarative": declarative,
            "has_hedge_phrase": has_hedge,
        }
        if hedges_found:
            details["hedge_phrases_found"] = hedges_found

        return self._result(score=score, details=details, recommendation=recommendation)

    def _recommend(
        self, word_count: int, declarative: bool, hedges: list[str]
    ) -> str | None:
        issues: list[str] = []
        if word_count > WORD_LIMIT:
            issues.append(
                f"Your opening paragraph is {word_count} words. Trim it to under "
                f"{WORD_LIMIT} words so AI engines can extract it as a direct answer."
            )
        if hedges:
            quoted = ", ".join(f'"{h}"' for h in hedges)
            issues.append(
                f"Remove hedge language ({quoted}) — answer engines prefer "
                "confident, specific statements."
            )
        if not declarative:
            issues.append(
                "Open with a complete declarative sentence (subject + verb) that "
                "states the answer, not a question or fragment."
            )
        return " ".join(issues) if issues else None
