"""Narrow signals for finding and checking explicitly requested comparisons."""

import re

from research_assist.domain.text import fold

_DIRECT = re.compile(
    r"\b(?:so(?:\s+sanh)?\s+truc\s+tiep|doi\s+chieu\s+truc\s+tiep|"
    r"(?:kha\s+nang|co\s+the)\s+(?:so\s+sanh|doi\s+chieu)|"
    r"cung\s+he\s+quy\s+chieu|"
    r"direct(?:ly)?\s+compar\w*|compar\w*(?:\s+\w+){0,8}\s+directly|"
    r"comparab(?:le|ility)|(?:in)?commensurab\w*|apples[- ]to[- ]apples)\b"
)
_PAPERS = re.compile(
    r"\b(?:(?:hai|2|cac|nhung)\s+bai\s+(?:bao|nghien\s+cuu)|"
    r"(?:two|both|these|different|across)\s+(?:papers|studies|articles)|cross[- ]paper)\b"
)
_RANKING = re.compile(
    r"\b(?:outperform(?:s|ed)?|outrank(?:s|ed)?|better than|worse than|higher than|lower than|"
    r"score[sd]?\s+higher|score[sd]?\s+lower|vuot\s+troi|tot\s+hon|kem\s+hon|"
    r"cao\s+hon|thap\s+hon)\b"
)
_NEGATIVE_CONCLUSION = re.compile(r"\b(?:not|never|cannot|can't|khong|chua|chang)\b")
_METRIC = re.compile(
    r"\b(?:bleu|rouge(?:-[12l])?|f1|accuracy|precision|recall|"
    r"do\s*chinh\s*xac|do\s*phu|perplexity|wer|cer|auc|rmse|mae)\b"
)
_MEASURED_NUMBER = re.compile(r"\b\d+(?:[.,]\d+|\s*%)")


def direct_comparison_clause(text: str) -> str | None:
    return next(
        (
            part.strip()
            for part in re.split(r"(?<=[?!.;])\s+|\n+", text)
            if _DIRECT.search(fold(part))
        ),
        None,
    )


def names_multiple_papers(text: str) -> bool:
    return bool(_PAPERS.search(fold(text)))


def states_comparison(text: str) -> bool:
    return bool(_DIRECT.search(fold(text)) or _RANKING.search(fold(text)))


def metric_names(text: str) -> set[str]:
    # PDF extraction can join Vietnamese label words inside a table header.
    names = {"".join(name.split()) for name in _METRIC.findall(fold(text))}
    if "dochinhxac" in names:
        names.remove("dochinhxac")
        names.add("accuracy")
    return names


def measured_metrics(text: str) -> set[str]:
    """Identify candidate windows, without asserting that a value belongs to a metric."""
    return metric_names(text) if _MEASURED_NUMBER.search(text) else set()


def ranks_distinct_metrics(text: str, requested: set[str]) -> bool:
    """Catch a literal positive ranking that puts a different named value on each side."""
    for sentence in re.split(r"(?<=[?!.;])\s+|\n+", fold(text)):
        rank = _RANKING.search(sentence)
        if not rank or _NEGATIVE_CONCLUSION.search(sentence):
            continue
        left = measured_metrics(sentence[: rank.start()]) & requested
        right = measured_metrics(sentence[rank.end() :]) & requested
        if len(left) == len(right) == 1 and left != right:
            return True
    return False
