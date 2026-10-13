"""Language-neutral tokenization for the lexical baseline (accent folding is intentional)."""

import re
import unicodedata


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
