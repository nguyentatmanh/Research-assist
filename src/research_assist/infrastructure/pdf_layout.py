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
    table: bool = False


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


def _aligned_rows(cells: list[tuple[Any, str]], left: float) -> str:
    """Keep blank cells and spanning headers in their printed horizontal positions."""
    rows: list[list[tuple[Any, str]]] = []
    for box, value in sorted(cells, key=lambda item: (item[0][3], item[0][0])):
        if not rows or abs(box[3] - rows[-1][0][0][3]) > 3:
            rows.append([])
        rows[-1].append((box, value))
    if sum(len(row) >= 3 for row in rows) < 2:
        return ""
    pitch = min((box[2] - box[0]) / len(value) for box, value in cells)
    if pitch < 1:
        return ""
    lines = []
    for row in rows:
        line = ""
        for box, value in sorted(row, key=lambda item: item[0][0]):
            column = round((box[0] - left) / pitch)
            line = line.ljust(max(column, len(line) + 1)) + value
        lines.append(line)
    return "\n".join(lines)


def _open_tables(
    page: Any, drawings: list[Any], raw_blocks: list[Any], grids: list[Block]
) -> list[Block]:
    """Limit borderless-column recovery to captioned tables with repeated full-width rules."""
    rails = [
        d["rect"]
        for d in drawings
        if d["rect"].height < 1 and d["rect"].width >= page.rect.width * 0.28
    ]
    if len(rails) < 3:
        return []
    cells = []
    captions = []
    for block in raw_blocks:
        lines = block.get("lines", [])
        for line in lines:
            pieces = []
            for span in line["spans"]:
                value = span["text"]
                if span["flags"] & 1:
                    value = value.translate(str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"))
                pieces.append(value)
            value = clean("".join(pieces))
            if value:
                cells.append((line["bbox"], value))
        if lines and re.match(
            r"^(Table|Bảng)\s+\d+[.:]", "".join(s["text"] for s in lines[0]["spans"]).strip(), re.I
        ):
            captions.append(block["bbox"])
    found: list[Block] = []
    for cap in captions:
        tops = [
            r for r in rails if -2 <= r.y0 - cap[3] <= 20 and min(r.x1, cap[2]) > max(r.x0, cap[0])
        ]
        if not tops:
            continue
        top = min(tops, key=lambda r: r.y0)
        matching = sorted(
            {
                r.y0
                for r in rails
                if r.y0 > top.y0 + 2 and abs(r.x0 - top.x0) < 2 and abs(r.x1 - top.x1) < 2
            }
        )
        bottom = top.y0
        count = 0
        for y in matching:
            if y - bottom > 90 or any(top.y0 < c[1] < y for c in captions):
                break
            bottom, count = y, count + 1
        if count < 2 or any(
            b.bbox[0] < top.x1 and b.bbox[2] > top.x0 and b.bbox[1] < bottom and b.bbox[3] > top.y0
            for b in grids + found
        ):
            continue
        items = [
            (box, value)
            for box, value in cells
            if top.y0 < (box[1] + box[3]) / 2 < bottom
            and top.x0 <= (box[0] + box[2]) / 2 <= top.x1
        ]
        if any(box[0] < top.x0 - 1 or box[2] > top.x1 + 1 for box, _ in items):
            continue
        body = _aligned_rows(items, top.x0)
        if body:
            found.append(Block(body, (top.x0, top.y0, top.x1, bottom), True))
    return found


def page_layout(page: Any, *, textpage: Any = None) -> tuple[str, list[SourceRegion]]:
    tables = []
    # Restrict table detection to ruled tables, avoiding guesses from columnar prose.
    drawings = page.get_drawings() if textpage is None else []
    if drawings:
        tables = list(page.find_tables(strategy="lines_strict").tables)
    blocks: list[Block] = []
    footnotes: list[Block] = []
    for table in tables:
        rows = [" | ".join(clean(cell or "") for cell in row) for row in table.extract()]
        text = "\n".join(rows).strip()
        if text:
            x0, y0, x1, y1 = table.bbox
            blocks.append(Block(text, (float(x0), float(y0), float(x1), float(y1)), True))
    raw_blocks = page.get_text("dict", textpage=textpage)["blocks"]
    blocks.extend(_open_tables(page, drawings, raw_blocks, blocks))
    table_boxes = [b.bbox for b in blocks]
    for raw in raw_blocks:
        if raw["type"] != 0:
            continue
        lines = []
        for line in raw["lines"]:
            x0, y0, x1, y1 = line["bbox"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if any(
                left <= cx <= right and top <= cy <= bottom
                for left, top, right, bottom in table_boxes
            ):
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
            SourceRegion(
                start=offset,
                end=offset + len(block.text),
                bbox=list(block.bbox),
                kind="table" if block.table else "text",
            )
        )
        offset += len(block.text)
    return "\n\n".join(parts), regions
