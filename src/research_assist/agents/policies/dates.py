"""Recognize explicit calendar dates without guessing numeric date locales."""

import re
from dataclasses import dataclass
from datetime import date

_MONTH_NAMES = (
    "january jan",
    "february feb",
    "march mar",
    "april apr",
    "may",
    "june jun",
    "july jul",
    "august aug",
    "september sept sep",
    "october oct",
    "november nov",
    "december dec",
)
_MONTHS = {name: month for month, names in enumerate(_MONTH_NAMES, 1) for name in names.split()}
_MONTH = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_PATTERNS = [
    re.compile(
        rf"(?<![\w+−-])(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+"
        rf"(?P<month>{_MONTH})\s*,?\s*(?P<year>\d{{4}})(?![\w%％])",
        re.I,
    ),
    re.compile(
        rf"(?<!\w)(?P<month>{_MONTH})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
        r"(?:\s*,\s*|\s+)(?P<year>\d{4})(?![\w%％])",
        re.I,
    ),
    re.compile(
        r"(?<![\w+−-])(?:ngày\s+)?(?P<day>\d{1,2})\s+tháng\s+(?P<month>\d{1,2})"
        r"\s+(?:năm\s+)?(?P<year>\d{4})(?![\w%％])",
        re.I,
    ),
    re.compile(r"(?<![\w/+−-])(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})(?![\w/%％-])"),
]
_NEGATED = re.compile(
    r"(?:\bnot|\bnever|rather than|instead of|không phải|không đạt)"
    r"(?:\s+(?:on|ngày))?\s*$",
    re.I,
)


@dataclass(frozen=True)
class DateMention:
    start: int
    end: int
    value: date | None
    negated: bool


def date_mentions(text: str) -> list[DateMention]:
    """Keep each date intact; never assemble one across separate source quotes.

    Slash dates are deliberately left to the literal guard: 04/11/2014 has no
    unambiguous month/day order without a locale supplied by the source.
    """
    mentions: list[DateMention] = []
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            if any(match.start() < m.end and m.start < match.end() for m in mentions):
                continue
            month_text = match["month"].lower().rstrip(".")
            month = int(month_text) if month_text.isdecimal() else _MONTHS[month_text]
            try:
                value = date(int(match["year"]), month, int(match["day"]))
            except ValueError:
                value = None
            mentions.append(
                DateMention(
                    match.start(),
                    match.end(),
                    value,
                    bool(_NEGATED.search(text[max(0, match.start() - 40) : match.start()])),
                )
            )
    return sorted(mentions, key=lambda m: m.start)


def without_dates(text: str, mentions: list[DateMention]) -> str:
    """Blank date spans while preserving offsets and all other numeric claims."""
    for mention in reversed(mentions):
        text = text[: mention.start] + " " * (mention.end - mention.start) + text[mention.end :]
    return text
