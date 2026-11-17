"""Cache paper discovery and merge results with their source information."""

import asyncio
import unicodedata
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic

from research_assist.infrastructure.arxiv import Paper
from research_assist.infrastructure.openalex import normalize_doi


@dataclass(frozen=True)
class _CachedPapers:
    expires_at: float
    papers: list[Paper]


class ScoutCache:
    """Bound successful results per provider and share concurrent identical lookups."""

    def __init__(self, *, ttl_seconds: int, max_entries: int = 256) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.entries: OrderedDict[tuple[str, str], _CachedPapers] = OrderedDict()
        self.pending: dict[tuple[str, str], asyncio.Task[list[Paper]]] = {}

    async def search(
        self, provider: str, query: str, fetch: Callable[[str], Awaitable[list[Paper]]]
    ) -> list[Paper]:
        doi = normalize_doi(query)
        normalized = doi or " ".join(query.split())
        if self.ttl_seconds <= 0 or self.max_entries <= 0:
            return await fetch(normalized)

        key = (provider, normalized)
        cached = self.entries.get(key)
        if cached is not None:
            if cached.expires_at > monotonic():
                self.entries.move_to_end(key)
                return [paper.model_copy(deep=True) for paper in cached.papers]
            del self.entries[key]

        task = self.pending.get(key)
        if task is None:
            ttl = self.ttl_seconds if doi else min(self.ttl_seconds, 3600)
            task = asyncio.create_task(self._populate(key, fetch, ttl))
            # A client may disconnect before a shared request finishes.
            task.add_done_callback(self._observe_result)
            self.pending[key] = task
        return [paper.model_copy(deep=True) for paper in await asyncio.shield(task)]

    @staticmethod
    def _observe_result(task: asyncio.Task[list[Paper]]) -> None:
        if not task.cancelled():
            task.exception()

    async def _populate(
        self, key: tuple[str, str], fetch: Callable[[str], Awaitable[list[Paper]]], ttl: int
    ) -> list[Paper]:
        try:
            papers = [paper.model_copy(deep=True) for paper in await fetch(key[1])]
            if not papers:
                ttl = min(ttl, 120)
            self.entries[key] = _CachedPapers(monotonic() + ttl, papers)
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
            return papers
        finally:
            self.pending.pop(key, None)

    async def close(self) -> None:
        outstanding = list(self.pending.values())
        for task in outstanding:
            task.cancel()
        await asyncio.gather(*outstanding, return_exceptions=True)
        self.pending.clear()


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _year(paper: Paper) -> int | None:
    value = paper.published[:4]
    return int(value) if len(value) == 4 and value.isascii() and value.isdigit() else None


def _same_title_and_identity(left: Paper, right: Paper) -> bool:
    if left.doi and right.doi:
        return False
    if _normalized_text(left.title) != _normalized_text(right.title):
        return False
    left_year, right_year = _year(left), _year(right)
    if left_year and right_year and left_year != right_year:
        return False
    left_authors = {_normalized_text(author) for author in left.authors if author.strip()}
    right_authors = {_normalized_text(author) for author in right.authors if author.strip()}
    if left_authors and right_authors:
        return bool(left_authors & right_authors)
    return left_year is not None and left_year == right_year


@dataclass
class _ScoutEntry:
    paper: Paper
    providers: list[str]
    arxiv_id: str | None

    def include(self, paper: Paper) -> None:
        if paper.source not in self.providers:
            self.providers.append(paper.source)
        if paper.source == "arxiv" and paper.import_available and self.arxiv_id is None:
            self.arxiv_id = paper.paper_id
            primary, secondary = paper, self.paper
        else:
            primary, secondary = self.paper, paper
        self.paper = primary.model_copy(
            update={
                "doi": primary.doi or secondary.doi,
                "authors": primary.authors or secondary.authors,
                "published": primary.published or secondary.published,
                "abstract": primary.abstract or secondary.abstract,
            }
        )


def merge_scout_papers(papers: list[Paper]) -> list[dict[str, object]]:
    entries: list[_ScoutEntry] = []
    for paper in papers:
        doi = normalize_doi(paper.doi) if paper.doi else None
        normalized = paper.model_copy(update={"doi": doi})
        match = next((entry for entry in entries if doi and entry.paper.doi == doi), None)
        if match is None:
            title_matches = [
                entry for entry in entries if _same_title_and_identity(entry.paper, normalized)
            ]
            if len(title_matches) == 1:
                match = title_matches[0]
        if match is not None:
            match.include(normalized)
            continue
        arxiv_id = paper.paper_id if paper.source == "arxiv" and paper.import_available else None
        entries.append(_ScoutEntry(normalized, [paper.source], arxiv_id))

    return [
        {
            **entry.paper.model_dump(),
            "provider": entry.paper.source,
            "providers": entry.providers,
            "arxiv_id": entry.arxiv_id,
            "import_available": entry.arxiv_id is not None,
            "year": _year(entry.paper),
        }
        for entry in entries
    ]
