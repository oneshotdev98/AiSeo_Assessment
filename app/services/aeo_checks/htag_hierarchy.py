"""Check B — H-tag Hierarchy.

Validates heading structure on the headings extracted in DOM order:
  1. Exactly one <h1>.
  2. No level is skipped going down (h1 -> h3 with no h2 is a violation).
  3. No heading appears before the first <h1>.

Plain-text input has no headings, which is reported as a missing H1 — the
content genuinely has no extractable heading structure for an answer engine.
"""

from __future__ import annotations

from app.models.schemas import CheckResult
from app.services.aeo_checks.base import BaseCheck
from app.services.content_parser import ParsedContent


def find_violations(h_tags: list[str]) -> list[str]:
    """Return human-readable violations for an ordered list like ['h1','h2','h3']."""
    violations: list[str] = []
    h1_count = h_tags.count("h1")

    if h1_count == 0:
        violations.append("Missing <h1>: the page has no top-level heading.")
    elif h1_count > 1:
        violations.append(
            f"Multiple <h1> tags found ({h1_count}); a page must have exactly one."
        )

    if h1_count >= 1:
        first_h1_index = h_tags.index("h1")
        for tag in h_tags[:first_h1_index]:
            violations.append(f"<{tag}> appears before the <h1>.")

    # Skipped levels: compare each heading to the previous one in DOM order.
    previous_level: int | None = None
    for tag in h_tags:
        level = int(tag[1])
        if previous_level is not None and level > previous_level + 1:
            violations.append(
                f"Heading level skipped: <h{previous_level}> is followed by "
                f"<{tag}> with no <h{previous_level + 1}> in between."
            )
        previous_level = level

    return violations


def score_for_violations(violations: list[str], has_h1: bool) -> int:
    if not has_h1:
        return 0
    if not violations:
        return 20
    if len(violations) <= 2:
        return 12
    return 0


class HtagHierarchyCheck(BaseCheck):
    check_id = "htag_hierarchy"
    name = "H-tag Hierarchy"

    def run(self, content: ParsedContent) -> CheckResult:
        h_tags = [tag for tag, _text in content.headings]
        violations = find_violations(h_tags)
        has_h1 = "h1" in h_tags
        score = score_for_violations(violations, has_h1)

        recommendation = None
        if violations:
            recommendation = (
                "Fix the heading structure: " + " ".join(violations) + " "
                "Use a single <h1> for the page title and nest sections "
                "H1 → H2 → H3 without skipping levels."
            )

        return self._result(
            score=score,
            details={"violations": violations, "h_tags_found": h_tags},
            recommendation=recommendation,
        )
