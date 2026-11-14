"""Bounded, read-only previews of the original source and a saved quotation."""

import hashlib
import math
import unicodedata
from pathlib import Path
from typing import Any, cast

from research_assist.config import Settings
from research_assist.domain.errors import DocumentError
from research_assist.domain.models import Document, Evidence


def render_region(path: Path, page_number: int, clip: str) -> bytes:
    import pymupdf

    fitz = cast(Any, pymupdf)

    try:
        coordinates = [float(v) for v in clip.split(",")]
        if len(coordinates) != 4 or not all(math.isfinite(v) for v in coordinates):
            raise ValueError("Invalid rectangle")
        x0, y0, x1, y1 = coordinates
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("Invalid rectangle")
        with fitz.open(path) as pdf:
            if not 1 <= page_number <= len(pdf):
                raise ValueError("Invalid page")
            page = pdf[page_number - 1]
            rect = fitz.Rect(coordinates)
            if not page.rect.contains(rect):
                raise ValueError("Rectangle outside page")
            context = fitz.Rect(x0 - 12, y0 - 12, x1 + 12, y1 + 12) & page.rect
            scale = min(2.0, 1100 / context.width, 1400 / context.height)
            page.draw_rect(rect, color=(0.95, 0.65, 0), width=1)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=context, alpha=False)
            return bytes(pix.tobytes("png"))
    except (ValueError, RuntimeError) as exc:
        raise DocumentError("Không thể mở vùng nguồn PDF; kiểm tra trang và tọa độ.") from exc


def _source_bytes(path: Path, digest: str, limit: int, verify: bool) -> bytes:
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except OSError as exc:
        raise DocumentError("Tệp gốc không còn đọc được.") from exc
    if not data or len(data) > limit:
        raise DocumentError("Tệp gốc rỗng hoặc vượt kích thước xem trước.")
    if verify and hashlib.sha256(data).hexdigest() != digest:
        raise DocumentError("Tệp gốc đã thay đổi so với bản được trích dẫn.")
    return data


def _normalized_quote(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _locate_quote(page: Any, evidence: Evidence, regions: list[Any], fitz: Any) -> list[Any]:
    """Accept a single exact reading of the quote inside its stored bounds."""
    quote = _normalized_quote(evidence.quote)
    if not quote or any(region.kind == "table" for region in evidence.regions):
        return []
    if not regions:
        return []
    clip = fitz.Rect()
    for rect in regions:
        clip.include_rect(rect)
    hits = [
        found
        for found in page.search_for(quote, clip=clip)
        if any(rect.contains(found) for rect in regions)
    ]
    if not hits or _normalized_quote(" ".join(page.get_textbox(hit) for hit in hits)) != quote:
        return []
    return hits


def render_page(
    path: Path, document: Document, page_number: int, evidence: Evidence | None, byte_limit: int
) -> tuple[bytes, str, float | None]:
    import pymupdf

    fitz = cast(Any, pymupdf)
    data = _source_bytes(path, document.sha256, byte_limit, verify=evidence is not None)
    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            if not 1 <= page_number <= len(pdf):
                raise ValueError("Invalid page")
            page = pdf[page_number - 1]
            bounds = page.rect * page.derotation_matrix
            highlight = "none"
            focus = None
            if evidence is not None:
                regions = [
                    rect
                    for item in evidence.regions
                    if not (rect := fitz.Rect(item.bbox) & bounds).is_empty
                ]
                exact = _locate_quote(page, evidence, regions, fitz)
                marks = exact or regions
                if marks:
                    highlight = "exact" if exact else "region"
                    for rect in marks:
                        page.draw_rect(
                            rect,
                            color=(0.70, 0.42, 0.0),
                            fill=(1.0, 0.84, 0.14) if exact else None,
                            fill_opacity=0.32,
                            width=0.9,
                        )
                    top = min((rect * page.rotation_matrix).y0 for rect in marks)
                    focus = max(0.0, min(1.0, (top - page.rect.y0) / page.rect.height))
            scale = min(2.5, 1800 / page.rect.width, 2400 / page.rect.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            return bytes(pix.tobytes("png")), highlight, focus
    except (ValueError, RuntimeError, ZeroDivisionError) as exc:
        raise DocumentError("Không thể xem trang PDF này.") from exc


def preview_text(
    path: Path, document: Document, evidence: Evidence | None, settings: Settings
) -> dict[str, object]:
    from research_assist.infrastructure.documents import extract

    data = _source_bytes(
        path, document.sha256, settings.max_upload_mb * 1024 * 1024, evidence is not None
    )
    (section,) = extract(data, document.filename, settings)
    start, end = (
        (evidence.start, evidence.end)
        if evidence is not None
        and 0 <= evidence.start < evidence.end <= len(section.text)
        and section.text[evidence.start : evidence.end] == evidence.quote
        else (None, None)
    )
    return {"text": section.text, "highlight_start": start, "highlight_end": end}
