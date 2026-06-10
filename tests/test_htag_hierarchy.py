"""Unit tests for Check B — H-tag Hierarchy."""

import pytest

from app.services.aeo_checks.htag_hierarchy import (
    HtagHierarchyCheck,
    find_violations,
    score_for_violations,
)
from app.services.content_parser import ParsedContent, parse_content


def make_content(h_tags: list[str]) -> ParsedContent:
    return ParsedContent(
        first_paragraph="Intro paragraph.",
        headings=[(tag, f"Heading {i}") for i, tag in enumerate(h_tags)],
        main_text="Some body text for the page.",
        is_html=True,
    )


@pytest.fixture
def check() -> HtagHierarchyCheck:
    return HtagHierarchyCheck()


class TestPassingCase:
    def test_valid_hierarchy_scores_full_marks(self, check):
        result = check.run(make_content(["h1", "h2", "h2", "h3", "h2"]))

        assert result.score == 20
        assert result.passed is True
        assert result.details["violations"] == []
        assert result.details["h_tags_found"] == ["h1", "h2", "h2", "h3", "h2"]
        assert result.recommendation is None


class TestFailingCases:
    def test_missing_h1_scores_0(self, check):
        result = check.run(make_content(["h2", "h3", "h2"]))

        assert result.score == 0
        assert result.passed is False
        assert any("Missing <h1>" in v for v in result.details["violations"])

    def test_single_skipped_level_scores_12(self, check):
        result = check.run(make_content(["h1", "h3", "h3"]))

        assert len(result.details["violations"]) == 1
        assert result.score == 12
        assert "skipped" in result.details["violations"][0].lower()

    def test_heading_before_h1_is_a_violation(self, check):
        result = check.run(make_content(["h2", "h1", "h2"]))

        assert any("before the <h1>" in v for v in result.details["violations"])
        assert result.score == 12

    def test_three_or_more_violations_scores_0(self, check):
        # multiple h1 + h2 before h1 + skipped level (h1 -> h4) = 3 violations
        result = check.run(make_content(["h2", "h1", "h4", "h1"]))

        assert len(result.details["violations"]) >= 3
        assert result.score == 0

    def test_plain_text_input_has_no_headings(self, check):
        content = ParsedContent(first_paragraph="Hello.", main_text="Hello there friend.")
        result = check.run(content)

        assert result.score == 0
        assert any("Missing <h1>" in v for v in result.details["violations"])


class TestDomOrderParsing:
    def test_headings_extracted_in_dom_order_even_inside_boilerplate(self):
        html = """
        <html><body>
          <header><h1>Page Title</h1></header>
          <article>
            <p>The opening paragraph of the article with enough words.</p>
            <h2>Section One</h2><p>Body text one.</p>
            <h3>Subsection</h3><p>Body text two.</p>
            <h2>Section Two</h2><p>Body text three.</p>
          </article>
          <footer><p>Copyright text in the footer area here.</p></footer>
        </body></html>
        """
        parsed = parse_content(html)
        assert [tag for tag, _ in parsed.headings] == ["h1", "h2", "h3", "h2"]
        assert find_violations([tag for tag, _ in parsed.headings]) == []


class TestScoring:
    @pytest.mark.parametrize(
        ("violations", "has_h1", "expected"),
        [
            ([], True, 20),
            (["one"], True, 12),
            (["one", "two"], True, 12),
            (["one", "two", "three"], True, 0),
            (["missing h1"], False, 0),
        ],
    )
    def test_score_table(self, violations, has_h1, expected):
        assert score_for_violations(violations, has_h1) == expected
