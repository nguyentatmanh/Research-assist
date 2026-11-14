"""Discover DOI metadata; discovery never adds unreviewed content to a research run."""

import asyncio
import json
import re
import time
from datetime import date
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field, ValidationError

from research_assist.domain.errors import CapacityError, DocumentError, ProviderError
from research_assist.infrastructure.arxiv import Paper, _download


class _Author(BaseModel):
    given: str = ""
    family: str = ""
    name: str = ""


class _Date(BaseModel):
    parts: list[list[int]] = Field(default_factory=list, alias="date-parts")

    def display(self) -> str:
        if not self.parts or not 1 <= len(self.parts[0]) <= 3:
            return ""
        parts = self.parts[0]
        try:
            value = date(
                parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1
            )
        except ValueError:
            return ""
        return value.isoformat()[: {1: 4, 2: 7, 3: 10}[len(parts)]]


class _Work(BaseModel):
    doi: str = Field(alias="DOI")
    title: list[str]
    author: list[_Author] = Field(default_factory=list)
    published: _Date = Field(default_factory=_Date)


class CrossrefLibrary:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self.searching = asyncio.Lock()
        self.next_search = 0.0

    async def search(self, query: str) -> list[Paper]:
        query = " ".join(query.split())
        if not 3 <= len(query) <= 300:
            raise DocumentError("Từ khóa tìm bài báo phải có từ 3 đến 300 ký tự.")
        if self.searching.locked() or time.monotonic() < self.next_search:
            raise CapacityError("Vui lòng đợi một giây giữa hai lần tìm trên Crossref.")
        async with self.searching:
            try:
                raw = await _download(
                    self.client,
                    "https://api.crossref.org/works",
                    524288,
                    {
                        "query.bibliographic": query,
                        "rows": "5",
                        "select": "DOI,title,author,published",
                    },
                    provider="Crossref",
                )
            finally:
                self.next_search = time.monotonic() + 1
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict) or payload.get("status") != "ok":
                raise ValueError("Expected a successful Crossref response")
            message = payload.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("items"), list):
                raise ValueError("Expected Crossref work items")
        except (UnicodeError, ValueError) as exc:
            raise ProviderError("Crossref trả về danh sách bài báo không hợp lệ.") from exc
        papers: list[Paper] = []
        seen: set[str] = set()
        for item in message["items"][:20]:
            try:
                work = _Work.model_validate(item)
            except ValidationError:
                continue
            doi = work.doi.strip().lower()
            title = " ".join(" ".join(work.title).split())
            if (
                not re.fullmatch(r"10\.\d{4,9}/[^\s\x00-\x1f\x7f]+", doi)
                or len(doi) > 256
                or doi in seen
                or not title
            ):
                continue
            seen.add(doi)
            papers.append(
                Paper(
                    paper_id="doi:" + doi,
                    title=title[:500],
                    authors=[
                        name[:200]
                        for author in work.author[:12]
                        if (
                            name := " ".join(
                                (author.name or f"{author.given} {author.family}").split()
                            )
                        )
                    ],
                    published=work.published.display(),
                    url="https://doi.org/" + quote(doi, safe="/"),
                    source="crossref",
                    doi=doi,
                    import_available=False,
                )
            )
            if len(papers) == 5:
                break
        return papers
