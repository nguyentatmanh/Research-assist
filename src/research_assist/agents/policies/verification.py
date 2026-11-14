"""Conservative structural guards; these are not a semantic entailment model."""

import re
from collections import Counter

from research_assist.agents.policies.dates import date_mentions, without_dates
from research_assist.agents.policies.quantities import unit_guard_reason
from research_assist.domain.models import Claim, ClaimCheck, Evidence
from research_assist.domain.provenance import evidence_source_hash
from research_assist.domain.text import fold

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
    return unit_guard_reason(claim, quotes)


_SIGNIFICANCE = re.compile(
    r"\b(?:statistically significant\b|statistical significance\b|y nghia thong ke\b)"
)
_NEGATIVE_SIGNIFICANCE = re.compile(
    r"\b(?:no|not|never|without|failed to|lack(?:ed|s)?|khong|chua)\b[^.;!?\n]{0,45}$"
)
_NEGATIVE_SIGNIFICANCE_AFTER = re.compile(
    r"^\s*(?:(?:was|is|were|has|have|had|could|can|did|do|does|been|be|duoc|da)\s+){0,3}"
    r"(?:not|never|khong|chua)\b"
)

# Keep statistical relations intact. A bag of numbers cannot distinguish p > .05
# from p < .05, nor an interval from two endpoints copied from unrelated quotes.
_STAT_VALUE = r"[+−-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
_P_STAT_VALUE = r"[+−-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][+-]?\d+)?"
_STAT_NEGATED = re.compile(fold(_NEGATED.pattern), re.I)
_P_VALUE = re.compile(
    rf"\bp(?:[ -]?values?)?\s*(?:(?:is|was|la)\s+)?"
    rf"(?P<op><=|>=|!=|[=<>≤≥≠≈])\s*(?P<value>{_P_STAT_VALUE})(?!\w|[.,]\d)",
    re.I,
)
_INTERVAL = re.compile(
    rf"(?:(?P<level>\d+(?:\.\d+)?)\s*%\s*)?"
    rf"\b(?:confidence\s+interval|ci|khoang\s+tin\s+cay)\b"
    rf"\s*(?:(?:is|was|la)\s+)?[:=]?\s*[\[(]\s*"
    rf"(?P<lower>{_STAT_VALUE})\s*[,;]\s*(?P<upper>{_STAT_VALUE})\s*[\])]",
    re.I,
)


def statistical_guard_reason(claim: str, quotes: list[str]) -> str:
    """Check explicit p relations and ordered confidence intervals, not significance.

    Deliberately no default alpha, inferred calculations, or semantic promotion.
    Entity, population, test selection and multiple-testing validity remain the
    semantic judge's responsibility. Only familiar explicit notation is parsed.
    """

    def p_values(text: str) -> set[tuple[str, str, bool]]:
        operators = {"<=": "≤", ">=": "≥", "!=": "≠"}
        text = fold(text)
        return {
            (
                operators.get(m["op"], m["op"]),
                m["value"],
                bool(_STAT_NEGATED.search(text[max(0, m.start() - 40) : m.start()])),
            )
            for m in _P_VALUE.finditer(text)
        }

    cited_p = set().union(*(p_values(q) for q in quotes))
    if not p_values(claim).issubset(cited_p):
        return "Quan hệ p-value không khớp với trích dẫn; không thể đổi dấu so sánh."

    intervals = [m.groupdict() for quote in quotes for m in _INTERVAL.finditer(fold(quote))]
    for mention in _INTERVAL.finditer(fold(claim)):
        if not any(
            mention["lower"] == row["lower"]
            and mention["upper"] == row["upper"]
            and (mention["level"] is None or mention["level"] == row["level"])
            for row in intervals
        ):
            return "Mức tin cậy hoặc cặp cận khoảng tin cậy không khớp trong cùng trích dẫn."
    return ""


def significance_guard_reason(claim: str, quotes: list[str]) -> str:
    """An explicit positive significance claim needs statistical evidence in its citations."""

    def positive(text: str) -> bool:
        folded = fold(text)
        return any(
            not _NEGATIVE_SIGNIFICANCE.search(folded[max(0, m.start() - 55) : m.start()])
            and not _NEGATIVE_SIGNIFICANCE_AFTER.search(folded[m.end() : m.end() + 55])
            for m in _SIGNIFICANCE.finditer(folded)
        )

    if positive(claim) and not any(positive(q) for q in quotes):
        return "Chưa có bằng chứng thống kê trong trích dẫn cho kết luận có ý nghĩa thống kê."
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
        elif any(
            lookup[key].source_hash
            and lookup[key].source_hash != evidence_source_hash(lookup[key])
            for key in check.evidence_ids
        ):
            reason = "Nội dung hoặc vị trí bằng chứng đã thay đổi so với mã băm nguồn đã đóng băng."
        elif check.verdict == "supported":
            quoted = [lookup[key].quote for key in check.evidence_ids]
            reason = (
                numeric_guard_reason(claim.text, quoted)
                or statistical_guard_reason(claim.text, quoted)
                or significance_guard_reason(claim.text, quoted)
            )
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
