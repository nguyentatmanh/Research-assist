"""Bounded, read-only rendering of a source rectangle from the original PDF."""

import math
from pathlib import Path
from typing import Any, cast

from research_assist.domain.errors import DocumentError


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
