"""Abstract base for AEO checks.

Each check is a self-contained unit: it receives ParsedContent and returns a
CheckResult. Adding a fourth check means writing one new subclass and adding
it to the registry in app/api/aeo.py — no other code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.schemas import CheckResult
from app.services.content_parser import ParsedContent


class BaseCheck(ABC):
    check_id: str
    name: str
    max_score: int = 20

    @abstractmethod
    def run(self, content: ParsedContent) -> CheckResult:
        """Score the content and return a fully-populated CheckResult."""

    def _result(
        self,
        score: int,
        details: dict,
        recommendation: str | None,
    ) -> CheckResult:
        return CheckResult(
            check_id=self.check_id,
            name=self.name,
            passed=score == self.max_score,
            score=score,
            max_score=self.max_score,
            details=details,
            recommendation=recommendation,
        )
