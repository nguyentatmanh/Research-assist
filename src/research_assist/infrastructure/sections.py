"""Conservative heading boundaries with a page/window fallback; never guesses printed pages."""

import re

from research_assist.domain.models import SourceSection

HEADING = re.compile(r"^(?:#{1,4}\s+|\d{1,2}(?:\.\d{1,2}){0,3}\.?\s+)[^\n]{3,90}$")


def split_headings(sections: list[SourceSection]) -> list[SourceSection]:
    result = []
    active = ""
    for source in sections:
        boundaries = []
        offset = 0
        for line in source.text.splitlines(keepends=True):
            clean = line.strip()
            if HEADING.match(clean) and not clean.endswith((".", ";")):
                boundaries.append((offset, clean.lstrip("# ")))
            offset += len(line)
        starts = [0, *[offset for offset, _ in boundaries if offset > 0]]
        headings = dict(boundaries)
        for i, start in enumerate(starts):
            active = headings.get(start, active)
            end = starts[i + 1] if i + 1 < len(starts) else len(source.text)
            if end > start:
                label = active or source.label
                if source.extraction == "ocr" and "OCR" not in label:
                    label += " · OCR"
                result.append(
                    source.model_copy(
                        update={
                            "text": source.text[start:end],
                            "offset": source.offset + start,
                            "label": label,
                        }
                    )
                )
    return result
