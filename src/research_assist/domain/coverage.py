"""Coverage refers to question requirements; a model judgement is not human gold."""

from research_assist.domain.models import (
    ClaimCheck,
    CoverageCheck,
    Evidence,
    EvidenceGap,
    GapCheck,
    Requirement,
    ResearchRequest,
)


def requirements_for(request: ResearchRequest) -> list[Requirement]:
    # User-authored aspects are authoritative. Do not pretend a heuristic understood a question.
    return [
        Requirement(id=f"rq_{i + 1}", text=text)
        for i, text in enumerate(request.requirements or [request.question])
    ]


def validate_coverage(
    requirements: list[Requirement], coverage: list[CoverageCheck], checks: list[ClaimCheck]
) -> list[CoverageCheck]:
    supported = {c.claim_id for c in checks if c.verdict == "supported"}
    result = []
    for requirement in requirements:
        rows = [c for c in coverage if c.requirement_id == requirement.id]
        valid = len(rows) == 1
        if valid:
            row = rows[0]
            valid = (
                len(row.claim_ids) == len(set(row.claim_ids))
                and set(row.claim_ids).issubset(supported)
                and (row.status not in {"covered", "partial"} or bool(row.claim_ids))
                and (row.status != "missing" or not row.claim_ids)
            )
        if valid:
            result.append(rows[0])
        else:
            result.append(
                CoverageCheck(
                    requirement_id=requirement.id,
                    status="unassessed",
                    explanation="Thiếu hoặc không hợp lệ đánh giá độ đầy đủ cho yêu cầu này.",
                )
            )
    return result


def report_coverage(coverage: list[CoverageCheck], written_ids: list[str]) -> list[CoverageCheck]:
    written = set(written_ids)
    result = []
    for row in coverage:
        retained = [cid for cid in row.claim_ids if cid in written]
        if row.status in {"covered", "partial"} and len(retained) != len(row.claim_ids):
            # Losing a cited claim invalidates the semantic assessment; don't infer coverage.
            result.append(
                CoverageCheck(
                    requirement_id=row.requirement_id,
                    status="unassessed",
                    claim_ids=retained,
                    explanation="Báo cáo chưa giữ đủ nhận định của đánh giá độ đầy đủ.",
                )
            )
        else:
            result.append(row)
    return result


def reconcile_gaps(
    requirements: list[Requirement],
    coverage: list[CoverageCheck],
    raw_gaps: list[str],
    gap_checks: list[GapCheck],
    evidence: list[Evidence],
) -> tuple[list[CoverageCheck], list[EvidenceGap]]:
    """Unvalidated free text never becomes a factual absence claim in a report.

    Missing/duplicate gap reviews fail closed. A resolved review must point to real evidence.
    An unassigned unresolved gap conservatively invalidates covered assessments.
    """
    evidence_ids = {e.id for e in evidence}
    valid_ids = {r.id for r in requirements}
    blocked = set()
    for index, _ in enumerate(raw_gaps):
        rows = [g for g in gap_checks if g.gap_index == index]
        if len(rows) == 1:
            row = rows[0]
            if (
                row.status == "resolved"
                and row.evidence_ids
                and set(row.evidence_ids).issubset(evidence_ids)
            ):
                continue
            related = set(row.requirement_ids) & valid_ids
            blocked.update(related or valid_ids)
        else:
            blocked.update(valid_ids)
    revised = []
    for coverage_row in coverage:
        if coverage_row.requirement_id in blocked and coverage_row.status == "covered":
            coverage_row = coverage_row.model_copy(
                update={
                    "status": "unassessed",
                    "explanation": (
                        "Còn khoảng trống chưa được giải quyết trong bằng chứng truy xuất."
                    ),
                }
            )
        revised.append(coverage_row)
    gaps = [
        EvidenceGap(
            requirement_id=row.requirement_id,
            reason="coverage_unassessed" if row.status == "unassessed" else "no_supported_evidence",
        )
        for row in revised
        if row.status != "covered"
    ]
    return revised, gaps


def gap_messages(
    gaps: list[EvidenceGap],
    requirements: list[Requirement],
    language: str,
) -> list[str]:
    lookup = {r.id: r.text for r in requirements}
    messages = []
    for gap in gaps:
        if gap.reason == "coverage_unassessed":
            prefix = (
                "Coverage has not been established for: "
                if language == "en"
                else "Chưa xác nhận được mức đáp ứng cho: "
            )
        else:
            prefix = (
                "The checked retrieved evidence does not yet fully answer: "
                if language == "en"
                else "Bằng chứng đã truy xuất và kiểm tra chưa trả lời đầy đủ: "
            )
        messages.append(prefix + lookup.get(gap.requirement_id or "", "unassessed aspect"))
    return messages
