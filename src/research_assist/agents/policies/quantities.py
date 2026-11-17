"""Check units and exact alternate spellings of explicitly labeled counts."""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from research_assist.domain.text import fold

_QUANTITY = re.compile(
    r"(?<![\w.,])(?P<value>[+−-]?\d+(?:[.,]\d+)*)\s*"
    r"(?P<unit>GiB|MiB|KiB|TiB|GB|MB|KB|TB|ms|s|kg|mg|g|mm|cm|km|m|%|％)"
    r"(?![\w])"
)

_SCALES = {
    "k": 3,
    "thousand": 3,
    "nghìn": 3,
    "ngàn": 3,
    "m": 6,
    "million": 6,
    "triệu": 6,
    "b": 9,
    "billion": 9,
    "tỷ": 9,
    "tỉ": 9,
}
_COUNT_UNITS = {
    "sample": "samples",
    "samples": "samples",
    "mẫu": "samples",
    "sentence": "sentences",
    "sentences": "sentences",
    "câu": "sentences",
    "example": "examples",
    "examples": "examples",
    "ví dụ": "examples",
    "question": "questions",
    "questions": "questions",
    "câu hỏi": "questions",
}
_COUNT = re.compile(
    r"(?<![\w.,])(?P<value>[+−-]?\d+(?:[.,]\d+)*)"
    rf"(?:[^\S\r\n]*(?P<scale>{'|'.join(_SCALES)}))?[^\S\r\n]+"
    rf"(?P<unit>{'|'.join(sorted(_COUNT_UNITS, key=len, reverse=True))})(?!\w)",
    re.I,
)
_COUNT_RELATION_BEFORE = re.compile(
    r"(?:(?P<at_most>\b(?:no more than|not more than|at most|up to|"
    r"khong qua|khong vuot qua|toi da|khong hon)\b|<=|≤)"
    r"|(?P<at_least>\b(?:no fewer than|no less than|not fewer than|not less than|at least|"
    r"it nhat|toi thieu|khong it hon|khong duoi|khong thap hon)\b|>=|≥)"
    r"|(?P<less_than>\b(?:fewer than|less than|under|below|it hon|duoi|nho hon)\b|<)"
    r"|(?P<more_than>\b(?:more than|greater than|over|above|nhieu hon|tren|lon hon|hon)\b|>)"
    r"|(?P<approximate>\b(?:approximately|approx\.?|about|around|roughly|nearly|"
    r"khoang|xap xi|gan|uoc chung|tam)(?!\w)|[~≈]))\s*$"
)
_COUNT_RELATION_AFTER = re.compile(
    r"^\s+(?:(?P<at_most>or fewer|or less|at most|tro xuong)"
    r"|(?P<at_least>or more|at least|tro len))\b"
)
_COUNT_NEGATED_BEFORE = re.compile(
    r"\b(?:not|never|no|rather than|instead of|khong|chua|"
    r"(?:don|doesn|didn|isn|aren|wasn|weren|haven|hasn|hadn)['’]t)\s*"
    r"(?:(?:he|da|duoc)\s+)?"
    r"(?:(?:us(?:e[ds]?|ing)|includ(?:e[ds]?|ing)|contain(?:s|ed)?|collect(?:s|ed)?|"
    r"have|had|analy[sz](?:e[ds]?|ing)|process(?:ed|es)?|"
    r"su dung|dung|co|dat|lay|thu thap|bao gom|phan tich|xu ly)\s+)?"
    r"(?:(?:exactly|precisely|all(?: of the)?|the full|chinh xac|dung|du)\s+)?$"
)
_COUNT_NEGATED_AFTER = re.compile(
    r"^\s+(?:(?:was|were|is|are|have been|has been|had been|da)\s+)?"
    r"(?:not|never|khong|chua|(?:wasn|weren|isn|aren)['’]t)\s+"
    r"(?:(?:used|included|collected|analy[sz]ed|processed|retained)|"
    r"(?:duoc\s+)?(?:su dung|dung|thu thap|phan tich|xu ly))\b"
)


@dataclass(frozen=True)
class CountMention:
    start: int
    end: int
    value: Decimal | tuple[str, int]
    unit: str
    scaled: bool
    relation: str
    negated: bool

    @property
    def requires_structured_match(self) -> bool:
        return self.scaled or self.relation != "exact" or self.negated


def count_mentions(text: str) -> list[CountMention]:
    counts = []
    for match in _COUNT.finditer(text):
        raw = match["value"].replace("−", "-")
        unit = match["unit"].lower()
        scale = match["scale"]
        if scale == "m":  # Lowercase m remains the SI length unit.
            continue
        power = _SCALES[scale.lower()] if scale else 0
        digits = _count_digits(raw, unit, scaled=bool(power))
        value = Decimal(f"{digits}e{power}") if digits is not None else (raw, power)
        relation, negated = _count_qualifiers(text, match.start(), match.end())
        counts.append(
            CountMention(
                start=match.start(),
                end=match.end(),
                value=value,
                unit=_COUNT_UNITS[unit],
                scaled=bool(power),
                relation=relation,
                negated=negated,
            )
        )
    return counts


def _count_qualifiers(text: str, start: int, end: int) -> tuple[str, bool]:
    before = re.sub(r"\s+", " ", fold(text[max(0, start - 100) : start]))
    after = re.sub(r"\s+", " ", fold(text[end : end + 65]))
    leading_relation = _COUNT_RELATION_BEFORE.search(before)
    relation = leading_relation or _COUNT_RELATION_AFTER.match(after)
    # Strip a leading bound before checking negation: "not more than" is one bound.
    if leading_relation:
        before = before[: leading_relation.start()]
    negated = bool(_COUNT_NEGATED_BEFORE.search(before) or _COUNT_NEGATED_AFTER.match(after))
    return (relation.lastgroup if relation else None) or "exact", negated


def _count_digits(raw: str, unit: str, *, scaled: bool) -> str | None:
    if "," not in raw and "." not in raw:
        return raw
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", raw) and (
        raw.count(",") > 1 or (unit.isascii() and not scaled)
    ):
        return raw.replace(",", "")
    if re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3}){2,}", raw):
        return raw.replace(".", "")
    # A single separator followed by three digits can also denote grouping.
    if scaled and re.fullmatch(r"[+-]?\d+[,.](?:\d{1,2}|\d{4,})", raw):
        return raw.replace(",", ".")
    return None


def quantities(text: str) -> set[tuple[str, str]]:
    result = set()
    for match in _QUANTITY.finditer(text):
        raw = match["value"].replace("−", "-")
        # Preserve comma notation literally so both decimal and grouping conventions
        # still bind a value to its unit without guessing a numeric conversion.
        try:
            value = raw if "," in raw else str(Decimal(raw).normalize())
            result.add((value, match["unit"].replace("％", "%")))
        except InvalidOperation:
            continue
    return result


def unit_guard_reason(claim: str, quotes: list[str]) -> str:
    expected = quantities(claim)
    observed = set().union(*(quantities(q) for q in quotes))
    if not expected.issubset(observed):
        return "Đại lượng hoặc đơn vị không khớp nguyên văn bằng chứng; cần kiểm tra lại."
    return ""
