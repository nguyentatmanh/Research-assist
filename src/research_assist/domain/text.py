"""Language-neutral tokenization for the lexical baseline (accent folding is intentional)."""

import re
import unicodedata
from collections.abc import Iterator


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


STOPWORDS = {
    fold(word)
    for word in """
    a an the and or of to in on for with is are was were be by from as it this that these those
    what how why does do can which about compare explain summarize research please
    và của là có một các những trong cho với được từ về này đó thì sẽ đã để không gì như nào
    tại sao hãy tôi bạn chúng giữa phân tích so sánh trình bày tổng quan
    """.split()
}


def tokens(text: str) -> list[str]:
    words = re.findall(r"[^\W_]+(?:[-_][^\W_]+)*", fold(text), re.UNICODE)
    return [word for word in words if len(word) > 1 and word not in STOPWORDS]


_BM25_WORDS = re.compile(
    r"(?<!\w)(?:"
    r"(?P<identifier>[a-z][^\W_]*(?:[-._][^\W_]+)+)(?!\w)"
    r"|(?P<number>[+\-]?\d+(?:[.,]\d+)*(?:[ \t]*%)?)(?!\w)"
    r"|(?P<word>[^\W_]+(?:[-_][^\W_]+)*))"
)
_PERCENT_HEADER = re.compile(r"(?:[(\[]\s*%\s*[)\]]|\s+%)$")
_TRANSLATIONS = {
    "mã hóa vị trí": "positional encoding",
    "làm mịn nhãn": "label smoothing",
    "chuẩn hóa lớp": "layer normalization",
    "tích vô hướng": "dot-product",
    "tự chú ý": "self-attention",
    "chú ý đa đầu": "multi-head attention",
    "mạng truyền thẳng": "feed-forward",
    "kết nối phần dư": "residual connection",
    "kết nối tàn dư": "residual connection",
    "che vị trí tương lai": "masking subsequent positions",
    "bộ mã hóa": "encoder",
    "bộ giải mã": "decoder",
}
_TRANSLATION_PHRASES = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(term) for term in _TRANSLATIONS) + r")(?!\w)"
)


def _bm25_words(text: str) -> Iterator[re.Match[str]]:
    return _BM25_WORDS.finditer(fold(text).replace("−", "-").replace("％", "%"))


def bm25_tokens(text: str) -> list[str]:
    """Index lexical literals while keeping the Reader's tokenizer independent."""
    result = []
    for match in _bm25_words(text):
        if number := match["number"]:
            number = number.replace(" ", "").replace("\t", "")
            result.append(number)
            if number.endswith("%"):
                result.append(number[:-1])
        else:
            word = match.group()
            if len(word) > 1 and word not in STOPWORDS:
                result.append(word)
    return result


def bilingual_query_groups(query: str) -> list[set[str]]:
    query = " ".join(unicodedata.normalize("NFC", query.casefold()).split())
    phrases = [match.group() for match in _TRANSLATION_PHRASES.finditer(query)]
    if not phrases:
        return []
    original = set(bm25_tokens(query))
    return [set(bm25_tokens(_TRANSLATIONS[phrase])) - original for phrase in phrases]


def extract_literal_anchors(text: str) -> set[str]:
    """Keep explicit identifiers, decimals and percentages, excluding plain counts."""
    anchors = set()
    for match in _bm25_words(text):
        if identifier := match["identifier"]:
            anchors.add(identifier)
        elif number := match["number"]:
            if any(marker in number for marker in ".,%"):
                anchors.add(number.replace(" ", "").replace("\t", ""))
    return anchors


def match_literal_anchors(anchors: set[str], text: str) -> set[str]:
    if not anchors:
        return set()
    matches = anchors & extract_literal_anchors(text)
    missing_percent = {anchor[:-1]: anchor for anchor in anchors - matches if anchor.endswith("%")}
    if not missing_percent or " | " not in text:
        return matches

    # PDF table rows use pipe delimiters. A bare value is a percent anchor only
    # in a column with an explicit percent header in the same source window.
    width = 0
    percent_columns: list[int] = []
    for line in text.replace("％", "%").splitlines():
        cells = [cell.strip() for cell in line.split(" | ")]
        if len(cells) != width:
            width, percent_columns = len(cells), []
        if width < 2:
            continue
        headers = [
            index
            for index, cell in enumerate(cells)
            if cell == "%" or (any(c.isalpha() for c in cell) and _PERCENT_HEADER.search(cell))
        ]
        if headers:
            percent_columns = headers
        else:
            for index in percent_columns:
                value = cells[index].replace("−", "-")
                if value in missing_percent:
                    matches.add(missing_percent[value])
    return matches
