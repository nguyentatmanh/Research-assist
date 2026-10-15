"""Coverage refers to question requirements; a model judgement is not human gold."""

from research_assist.domain.models import ClaimCheck, CoverageCheck, Requirement, ResearchRequest


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
