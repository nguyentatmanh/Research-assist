"""Reject explicit unit mismatches; never establish semantic entailment.

Unlike the upstream tolerance verifier, 5 is not a substring match for 50,
and 5 MB is not interchangeable with 5 GB. Conversions require an explicitly
derived claim/tool in a future version; this guard conservatively abstains.
"""

import re
from decimal import Decimal, InvalidOperation

_QUANTITY = re.compile(
    r"(?<![\w.,])(?P<value>[+−-]?\d+(?:[.,]\d+)*)\s*"
    r"(?P<unit>GiB|MiB|KiB|TiB|GB|MB|KB|TB|ms|s|kg|mg|g|mm|cm|km|m|%|％)"
    r"(?![\w])"
)


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
