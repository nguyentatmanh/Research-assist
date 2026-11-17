"""Bind exact normalized-page evidence to its source coordinates."""

import hashlib
import json

from research_assist.domain.models import Evidence, SourceRegion


def evidence_source_hash(evidence: Evidence) -> str:
    """Hash immutable content/coordinates; document versions are checked at run commit."""
    payload = [
        evidence.document_id,
        evidence.chunk_id,
        evidence.page,
        evidence.start,
        evidence.end,
        evidence.quote,
        [region.model_dump() for region in evidence.regions],
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def intersect_regions(regions: list[SourceRegion], start: int, end: int) -> list[SourceRegion]:
    return [
        r.model_copy(update={"start": max(start, r.start), "end": min(end, r.end)})
        for r in regions
        if r.start < end and start < r.end
    ]
