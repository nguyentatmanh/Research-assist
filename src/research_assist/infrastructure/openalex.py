"""Discover paper metadata from the fixed OpenAlex Works API."""

import asyncio
import json
import re
import time
from datetime import date
from urllib.parse import quote, unquote, urlsplit

import httpx

from research_assist.domain.errors import CapacityError, DocumentError, NotFoundError, ProviderError
from research_assist.infrastructure.arxiv import Paper, _download

_WORKS = "https://api.openalex.org/works"
_WORK_ID = re.compile(r"https://openalex\.org/(W[0-9]{1,20})", re.ASCII | re.I)
_DOI = re.compile(r"10\.[0-9]{4,9}/[^\s\x00-\x1f\x7f]+")
_DOI_URL = re.compile(r"https?://(?:dx\.)?doi\.org/", re.ASCII | re.I)
_SURROGATE = re.compile(r"[\ud800-\udfff]")
_FIELDS = "id,title,doi,publication_date,publication_year,authorships,abstract_inverted_index"
_MAX_RESULTS = 5
_ABSTRACT_WORDS = 2048
_ABSTRACT_CHARS = 16000


def normalize_doi(value: str) -> str | None:
    doi = value.strip()
    if _DOI_URL.match(doi):
        doi = unquote(urlsplit(doi).path.removeprefix("/"))
    elif doi[:4].lower() == "doi:":
        doi = doi[4:].strip()
    if len(doi) > 256 or not _DOI.fullmatch(doi) or _SURROGATE.search(doi):
        return None
    # URLs normalize dot segments when a DOI is put in the path.
    if any(segment in {".", ".."} for segment in doi.split("/")):
        return None
    return doi.lower()


def _abstract(index: object) -> str | None:
    if not isinstance(index, dict) or not 1 <= len(index) <= _ABSTRACT_WORDS:
        return None
    words: dict[int, str] = {}
    length = 0
    for word, positions in index.items():
        if not isinstance(word, str) or not word.strip() or not isinstance(positions, list):
            return None
        if _SURROGATE.search(word):
            return None
        if not positions or len(positions) + len(words) > _ABSTRACT_WORDS:
            return None
        length += len(positions) * (len(word) + 1)
        if length - 1 > _ABSTRACT_CHARS:
            return None
        for position in positions:
            if type(position) is not int or not 0 <= position < _ABSTRACT_WORDS:
                return None
            if position in words:
                return None
            words[position] = word
    if len(words) != max(words) + 1:
        return None
    return " ".join(words[position] for position in range(len(words)))


def _authors(authorships: object) -> list[str]:
    if not isinstance(authorships, list):
        return []
    authors = []
    for item in authorships[:12]:
        author = item.get("author") if isinstance(item, dict) else None
        name = author.get("display_name") if isinstance(author, dict) else None
        if (
            isinstance(name, str)
            and not _SURROGATE.search(name)
            and (name := " ".join(name.split()))
        ):
            authors.append(name[:200])
    return authors


def _published(value: object, year: object) -> str:
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            pass
    return f"{year:04d}" if type(year) is int and 1 <= year <= 9999 else ""


def _paper(work: object) -> Paper | None:
    if not isinstance(work, dict):
        return None
    identifier, title = work.get("id"), work.get("title")
    match = _WORK_ID.fullmatch(identifier) if isinstance(identifier, str) else None
    if match is None or not isinstance(title, str) or _SURROGATE.search(title):
        return None
    if not (title := " ".join(title.split())):
        return None
    work_id = match[1].upper()
    raw_doi = work.get("doi")
    doi = normalize_doi(raw_doi) if isinstance(raw_doi, str) else None
    return Paper(
        paper_id=work_id,
        title=title[:500],
        authors=_authors(work.get("authorships")),
        published=_published(work.get("publication_date"), work.get("publication_year")),
        url=f"https://doi.org/{quote(doi, safe='/')}" if doi else f"https://openalex.org/{work_id}",
        source="openalex",
        doi=doi,
        import_available=False,
        abstract=_abstract(work.get("abstract_inverted_index")),
    )


class OpenAlexLibrary:
    def __init__(self, client: httpx.AsyncClient, api_key: str = "") -> None:
        self.client = client
        self.headers = {"Authorization": f"Bearer {api_key.strip()}"} if api_key.strip() else {}
        self.searching = asyncio.Lock()
        self.next_search = 0.0

    async def search(self, query: str) -> list[Paper]:
        query = " ".join(query.split())
        if not 3 <= len(query) <= 300:
            raise DocumentError("Từ khóa tìm bài báo phải có từ 3 đến 300 ký tự.")
        if self.searching.locked() or time.monotonic() < self.next_search:
            raise CapacityError("Vui lòng đợi một giây giữa hai lần tìm trên OpenAlex.")
        doi = normalize_doi(query)
        url = f"{_WORKS}/doi:{quote(doi, safe='/')}" if doi else _WORKS
        params = {"select": _FIELDS}
        if doi is None:
            params.update(search=query, per_page=str(_MAX_RESULTS))
        async with self.searching:
            try:
                raw = await _download(
                    self.client,
                    url,
                    524288,
                    params,
                    provider="OpenAlex",
                    headers=self.headers,
                    timeout_seconds=8,
                    allow_not_found=doi is not None,
                )
            except NotFoundError:
                return []
            finally:
                self.next_search = time.monotonic() + 1
        try:
            payload = json.loads(raw)
            if doi is not None:
                paper = _paper(payload)
                if paper is None:
                    raise ValueError("Expected a Work record")
                return [paper]
            results = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(results, list):
                raise ValueError("Expected a Works result list")
        except (UnicodeError, ValueError, RecursionError) as exc:
            raise ProviderError("OpenAlex trả về dữ liệu bài báo không hợp lệ.") from exc
        return list(
            {
                paper.paper_id: paper
                for result in results[:_MAX_RESULTS]
                if (paper := _paper(result)) is not None
            }.values()
        )
