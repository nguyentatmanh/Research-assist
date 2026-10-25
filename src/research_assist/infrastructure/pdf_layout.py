"""Order PDF text by whitespace-separated columns and preserve page rectangles."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from research_assist.domain.models import SourceRegion


@dataclass
class Block:
    text: str
    bbox: tuple[float, float, float, float]


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _cut(blocks: list[Block], axis: int) -> tuple[float, float]:
    intervals = sorted((b.bbox[axis], b.bbox[axis + 2]) for b in blocks)
    end = intervals[0][1]
    best = (0.0, 0.0)
    for left, right in intervals[1:]:
        if left - end > best[0]:
            best = (left - end, (left + end) / 2)
        end = max(end, right)
    return best


def reading_order(blocks: list[Block], depth: int = 0) -> list[Block]:
    """Recursive XY cuts: isolate spanning headers, then read each column top to bottom."""
    if len(blocks) < 2:
        return blocks
    if depth >= 32:
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    width = max(b.bbox[2] for b in blocks) - min(b.bbox[0] for b in blocks)
    for anchor in sorted(blocks, key=lambda b: b.bbox[1]):
        if anchor.bbox[2] - anchor.bbox[0] < width * 0.8:
            continue
        before = [b for b in blocks if b is not anchor and b.bbox[3] <= anchor.bbox[1]]
        after = [b for b in blocks if b is not anchor and b.bbox[1] >= anchor.bbox[3]]
        if len(before) + len(after) == len(blocks) - 1:
            return reading_order(before, depth + 1) + [anchor] + reading_order(after, depth + 1)
    xgap, xcut = _cut(blocks, 0)
    ygap, ycut = _cut(blocks, 1)
    if max(xgap, ygap) < 4:
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    # A clear gutter takes priority over paragraph spacing inside the columns.
    axis, cut = (0, xcut) if xgap >= 12 or ygap < 4 else (1, ycut)
    first = [b for b in blocks if b.bbox[axis + 2] <= cut]
    second = [b for b in blocks if b.bbox[axis] >= cut]
    if not first or not second or len(first) + len(second) != len(blocks):
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    return reading_order(first, depth + 1) + reading_order(second, depth + 1)


def _line_columns(lines: list[Block]) -> list[Block]:
    gap, cut = _cut(lines, 0)
    if gap >= 12:
        left = [b for b in lines if b.bbox[2] <= cut]
        right = [b for b in lines if b.bbox[0] >= cut]
        return _line_columns(left) + _line_columns(right)
    return [
        Block(
            "\n".join(b.text for b in sorted(lines, key=lambda b: (b.bbox[1], b.bbox[0]))),
            (
                min(b.bbox[0] for b in lines),
                min(b.bbox[1] for b in lines),
                max(b.bbox[2] for b in lines),
                max(b.bbox[3] for b in lines),
            ),
        )
    ]


def page_layout(page: Any, *, textpage: Any = None) -> tuple[str, list[SourceRegion]]:
    tables = []
    # Restrict table detection to ruled tables, avoiding guesses from columnar prose.
    if textpage is None and page.get_drawings():
        tables = list(page.find_tables(strategy="lines_strict").tables)
    blocks: list[Block] = []
    footnotes: list[Block] = []
    for table in tables:
        rows = [" | ".join(clean(cell or "") for cell in row) for row in table.extract()]
        text = "\n".join(rows).strip()
        if text:
            x0, y0, x1, y1 = table.bbox
            blocks.append(Block(text, (float(x0), float(y0), float(x1), float(y1))))
    for raw in page.get_text("dict", textpage=textpage)["blocks"]:
        if raw["type"] != 0:
            continue
        lines = []
        for line in raw["lines"]:
            x0, y0, x1, y1 = line["bbox"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if any(t.bbox[0] <= cx <= t.bbox[2] and t.bbox[1] <= cy <= t.bbox[3] for t in tables):
                continue
            text = clean("".join(span["text"] for span in line["spans"]))
            if text:
                lines.append((text, line["bbox"]))
        if lines:
            bbox = (
                min(item[1][0] for item in lines),
                min(item[1][1] for item in lines),
                max(item[1][2] for item in lines),
                max(item[1][3] for item in lines),
            )
            groups = _line_columns(
                [Block(text, (box[0], box[1], box[2], box[3])) for text, box in lines]
            )
            small = max(span["size"] for line in raw["lines"] for span in line["spans"]) < 8
            for block in groups:
                if small and (bbox[1] > page.rect.height * 0.83 or block.text.startswith("©")):
                    footnotes.append(block)
                else:
                    blocks.append(block)
    ordered = reading_order(blocks) + reading_order(footnotes)
    merged: list[Block] = []
    for block in ordered:
        # Restore a typographic drop cap only when its immediately adjacent prose
        # starts with a lower-case letter at the same column origin.
        if (
            merged
            and re.fullmatch(r"[A-Z]", merged[-1].text)
            and block.text[:1].islower()
            and abs(merged[-1].bbox[0] - block.bbox[0]) < 8
            and abs(merged[-1].bbox[1] - block.bbox[1]) < 24
        ):
            cap = merged.pop()
            block = Block(
                cap.text + block.text,
                (
                    min(cap.bbox[0], block.bbox[0]),
                    min(cap.bbox[1], block.bbox[1]),
                    max(cap.bbox[2], block.bbox[2]),
                    max(cap.bbox[3], block.bbox[3]),
                ),
            )
        merged.append(block)
    parts: list[str] = []
    regions: list[SourceRegion] = []
    offset = 0
    for block in merged:
        if parts:
            offset += 2
        parts.append(block.text)
        regions.append(
            SourceRegion(start=offset, end=offset + len(block.text), bbox=list(block.bbox))
        )
        offset += len(block.text)
    return "\n\n".join(parts), regions
