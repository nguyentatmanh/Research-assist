"""Read public arXiv metadata and versioned PDFs from fixed HTTPS endpoints."""

import asyncio
import re
import time
from urllib.parse import urlsplit
from xml.etree import ElementTree

import httpx
from pydantic import BaseModel

from research_assist.domain.errors import CapacityError, DocumentError, ProviderError

PAPER_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z][a-z.-]*/\d{7})v[1-9]\d{0,3}", re.ASCII)
ATOM = "{http://www.w3.org/2005/Atom}"


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    published: str
    url: str


async def _download(
    client: httpx.AsyncClient, url: str, limit: int, params: dict[str, str] | None = None
) -> bytes:
    try:
        async with (
            asyncio.timeout(45),
            client.stream(
                "GET",
                url,
                params=params,
                timeout=20,
                follow_redirects=False,
                headers={"User-Agent": "ResearchDesk/1.0 (academic paper discovery)"},
            ) as response,
        ):
            if response.status_code == 429:
                raise CapacityError("arXiv đang giới hạn lượt truy cập. Hãy thử lại sau.")
            response.raise_for_status()
            data = bytearray()
            async for part in response.aiter_bytes(65536):
                if len(data) + len(part) > limit:
                    raise DocumentError("Dữ liệu từ arXiv vượt giới hạn dung lượng.")
                data.extend(part)
            return bytes(data)
    except (httpx.HTTPError, TimeoutError) as exc:
        raise ProviderError("Không nhận được phản hồi hợp lệ từ arXiv. Hãy thử lại sau.") from exc


class ArxivLibrary:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self.next_search = 0.0
        self.searching = asyncio.Lock()
        self.recent: dict[str, Paper] = {}

    async def search(self, query: str) -> list[Paper]:
        query = " ".join(query.split())
        if len(query) < 3:
            raise DocumentError("Nhập ít nhất ba ký tự để tìm bài báo.")
        if self.searching.locked() or time.monotonic() < self.next_search:
            raise CapacityError("Vui lòng đợi ba giây giữa hai lần tìm trên arXiv.")
        # Serialize API requests, including overlapping searches from different browser tabs.
        async with self.searching:
            try:
                raw = await _download(
                    self.client,
                    "https://export.arxiv.org/api/query",
                    524288,
                    {
                        "search_query": f"all:{query}",
                        "sortBy": "relevance",
                        "max_results": "5",
                    },
                )
            finally:
                self.next_search = time.monotonic() + 3
        try:
            xml = raw.decode("utf-8-sig")
            if "<!DOCTYPE" in xml or "<!ENTITY" in xml:
                raise ValueError("Unexpected declarations in Atom feed")
            root = ElementTree.fromstring(xml)
            if root.tag != ATOM + "feed":
                raise ValueError("Expected an Atom feed")
            papers = []
            for entry in root.findall(ATOM + "entry"):
                identifier = urlsplit(entry.findtext(ATOM + "id", ""))
                if identifier.hostname != "arxiv.org" or not identifier.path.startswith("/abs/"):
                    raise ValueError("Unexpected entry in Atom feed")
                paper_id = identifier.path.removeprefix("/abs/")
                title = " ".join(entry.findtext(ATOM + "title", "").split())
                if not PAPER_ID.fullmatch(paper_id) or not title:
                    continue
                papers.append(
                    Paper(
                        paper_id=paper_id,
                        title=title[:500],
                        authors=[
                            n.text
                            for n in entry.findall(ATOM + "author/" + ATOM + "name")
                            if n.text
                        ][:12],
                        published=entry.findtext(ATOM + "published", "")[:10],
                        url=f"https://arxiv.org/abs/{paper_id}",
                    )
                )
            for paper in papers[:5]:
                self.recent.pop(paper.paper_id, None)
                self.recent[paper.paper_id] = paper
            while len(self.recent) > 100:
                del self.recent[next(iter(self.recent))]
            return papers[:5]
        except (UnicodeError, ElementTree.ParseError, ValueError) as exc:
            raise ProviderError("arXiv trả về danh sách bài báo không hợp lệ.") from exc

    async def pdf(self, paper_id: str, limit: int) -> bytes:
        if not PAPER_ID.fullmatch(paper_id):
            raise DocumentError("Mã arXiv phải gồm số phiên bản đã chọn.")
        data = await _download(self.client, f"https://arxiv.org/pdf/{paper_id}", limit)
        if not data.lstrip().startswith(b"%PDF-"):
            raise DocumentError("arXiv chưa cung cấp tệp PDF cho phiên bản này.")
        return data
