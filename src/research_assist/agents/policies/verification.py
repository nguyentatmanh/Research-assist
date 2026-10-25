"""Conservative structural guards; these are not a semantic entailment model."""

import re
from collections import Counter

from research_assist.agents.policies.dates import date_mentions, without_dates
from research_assist.domain.models import Claim, ClaimCheck, Evidence

_NUMBER = re.compile(r"(?<!\w)[+−-]?\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?(?:%|％)?")
_LIST_MARKER = re.compile(r"(?m)(?<!\S)(\d{1,2})[.)](?=\s+\S)")
_NEGATED = re.compile(r"(?:\bnot|\bnever|rather than|instead of|không phải|không đạt)\s*$", re.I)


def _without_list_markers(text: str) -> str:
    """Presentation ordinals are not numerical research claims."""
    markers = list(_LIST_MARKER.finditer(text))
    # A lone number followed by a full stop can be a measurement at sentence end.
    # Only a complete consecutive sequence starting at one is a presentation list.
    if len(markers) < 2 or [int(m.group(1)) for m in markers] != list(range(1, len(markers) + 1)):
        return text
    prefix = text[: markers[0].start()].rsplit("\n", 1)[-1].rstrip()
    if prefix and not prefix.endswith(":"):
        return text
    for marker in reversed(markers):
        text = text[: marker.start()] + text[marker.end() :]
    return text


def numbers(text: str) -> set[str]:
    return {match.group() for match in _NUMBER.finditer(_without_list_markers(text))}


def number_mentions(text: str) -> set[tuple[str, bool]]:
    text = _without_list_markers(text)
    return {
        (m.group(), bool(_NEGATED.search(text[max(0, m.start() - 40) : m.start()])))
        for m in _NUMBER.finditer(text)
    }


def numeric_guard_reason(claim: str, quotes: list[str]) -> str:
    """Check explicit dates as whole values, then retain the literal numeric guards.

    This never establishes semantic support or changes a model's rejection.
    Normalized month numbers cannot be borrowed to support unrelated measurements.
    """
    dates = date_mentions(claim)
    if any(d.value is None for d in dates):
        return "Nhận định chứa ngày tháng không hợp lệ."
    if dates:
        source_dates = {
            (d.value, d.negated)
            for quote in quotes
            for d in date_mentions(quote)
            if d.value is not None
        }
        if not {(d.value, d.negated) for d in dates}.issubset(source_dates):
            return "Ngày tháng hoặc dấu phủ định không khớp với các trích dẫn đã kiểm tra."
    remaining = without_dates(claim, dates)
    if not numbers(remaining).issubset(numbers(" ".join(quotes))):
        return "Có con số không xuất hiện trong các trích dẫn đã kiểm tra."
    if not number_mentions(remaining).issubset(set().union(*(number_mentions(q) for q in quotes))):
        return "Con số có dấu phủ định khác với bằng chứng; cần kiểm chứng lại."
    return ""


def validate_checks(
    claims: list[Claim], evidence: list[Evidence], checks: list[ClaimCheck]
) -> list[ClaimCheck]:
    evidence_counts = Counter(e.id for e in evidence)
    lookup = {e.id: e for e in evidence if evidence_counts[e.id] == 1}
    claim_counts = Counter(c.id for c in claims)
    groups: dict[str, list[ClaimCheck]] = {}
    for incoming in checks:
        groups.setdefault(incoming.claim_id, []).append(incoming)
    validated = []
    for claim in claims:
        candidates = groups.get(claim.id, [])
        reason = ""
        check = candidates[0] if len(candidates) == 1 else None
        if claim_counts[claim.id] != 1 or check is None:
            reason = "Thiếu hoặc trùng nhận định/kết quả kiểm tra."
        elif not check.evidence_ids or any(
            key not in lookup or key not in claim.evidence_ids for key in check.evidence_ids
        ):
            reason = "Kết quả kiểm tra không trỏ tới bằng chứng hợp lệ của nhận định."
        elif check.verdict == "supported":
            quoted = [lookup[key].quote for key in check.evidence_ids]
            reason = numeric_guard_reason(claim.text, quoted)
        if reason or check is None:
            validated.append(
                ClaimCheck(
                    claim_id=claim.id, verdict="insufficient", evidence_ids=[], explanation=reason
                )
            )
        else:
            validated.append(
                check.model_copy(update={"evidence_ids": list(dict.fromkeys(check.evidence_ids))})
            )
    return validated
