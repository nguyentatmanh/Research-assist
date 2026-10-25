"""Intersect normalized-page text offsets without inventing PDF coordinates."""

from research_assist.domain.models import SourceRegion


def intersect_regions(regions: list[SourceRegion], start: int, end: int) -> list[SourceRegion]:
    return [
        r.model_copy(update={"start": max(start, r.start), "end": min(end, r.end)})
        for r in regions
        if r.start < end and start < r.end
    ]
