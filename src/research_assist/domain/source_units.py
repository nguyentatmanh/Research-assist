"""Exact extractive windows. Coordinates refer to normalized extracted page text."""

import re
from hashlib import sha256

from research_assist.domain.models import RetrievedChunk


def source_units(chunks: list[RetrievedChunk]) -> list[dict[str, str | int]]:
    units: list[dict[str, str | int]] = []
    for chunk in chunks:
        start = 0
        while start < len(chunk.text):
            end = min(start + 850, len(chunk.text))
            if end < len(chunk.text):
                matches = list(re.finditer(r"(?<=[.!?])\s+|\n\n", chunk.text[start + 250 : end]))
                if matches:
                    end = start + 250 + matches[-1].start()
            # Keep a short final sentence with its context, without creating an oversized quote.
            if len(chunk.text) - start <= 869 and len(chunk.text[end:].strip()) < 20:
                end = len(chunk.text)
            raw = chunk.text[start:end]
            left = start + len(raw) - len(raw.lstrip())
            text = raw.strip()
            # Short standalone text can still carry a result or a negation. Never discard it.
            if text:
                uid = "su_" + sha256(f"{chunk.id}:{left}:{text}".encode()).hexdigest()[:20]
                units.append(
                    {
                        "unit_id": uid,
                        "chunk_id": chunk.id,
                        "start": left,
                        "end": left + len(text),
                        "text": text,
                    }
                )
            if end >= len(chunk.text):
                break
            start = end
    return units
