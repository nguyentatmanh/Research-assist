"""Pure, bounded transition rules. Only the application executes transitions."""

from dataclasses import dataclass

from research_assist.domain.models import ClaimCheck, CoverageCheck


@dataclass(frozen=True)
class Decision:
    retrieve_again: bool
    reason: str


def decide_next(
    *,
    round_number: int,
    max_rounds: int,
    chunk_count: int,
    new_evidence_count: int,
    checks: list[ClaimCheck],
    coverage: list[CoverageCheck],
    gaps: list[str],
) -> Decision:
    if not chunk_count:
        return Decision(False, "no_retrieval_results")
    if round_number > 1 and not new_evidence_count:
        return Decision(False, "no_new_evidence")
    supported = any(c.verdict == "supported" for c in checks)
    unresolved = bool(gaps) or any(c.verdict != "supported" for c in checks)
    unresolved |= any(c.status in {"missing", "partial"} for c in coverage)
    if supported and not unresolved:
        complete = bool(coverage) and all(c.status == "covered" for c in coverage)
        return Decision(False, "requirements_covered" if complete else "coverage_unassessed")
    if round_number >= max_rounds:
        return Decision(False, "round_limit")
    return Decision(True, "retrieve_missing_evidence")
