import json
from contextlib import contextmanager

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient

from research_assist.agents.policies.verification import validate_checks
from research_assist.api.app import create_app
from research_assist.bootstrap import build_container
from research_assist.domain.errors import DocumentError, ProviderError
from research_assist.domain.models import Claim, ClaimCheck
from research_assist.experiments.system_benchmark import load_cases
from research_assist.infrastructure.arxiv import ArxivLibrary


def table_pdf() -> bytes:
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((43, 57), "Table 1. Measured performance")
        for y in (64, 90, 148):
            page.draw_line((40, y), (400, y))
        for y, row in [
            (82, ("Method", "Group One", "Group Two")),
            (108, ("Reference", "82.1", "91.7")),
            (128, ("Candidate", "", "95.4")),
        ]:
            for x, cell in zip((43, 170, 300), row, strict=True):
                if cell:
                    page.insert_text((x, y), cell)
        page = doc.new_page()
        page.insert_text((40, 50), "Separate results on a different physical page.")
        return doc.tobytes()


async def test_table_windows_are_pinned_and_preserve_blank_columns(settings):
    container = build_container(settings)
    try:
        doc, _ = await container.documents.ingest(table_pdf(), "study.pdf")
        frozen = await container.search.freeze([doc.id])
        seeds, _ = await frozen.search(["Table performance"], [doc.id], 3, "lexical")
        await container.documents.delete(doc.id)
        tables = await frozen.read_tables(seeds, [doc.id])
        assert tables and all(c.page == 1 and c.document_id == doc.id for c in tables)
        region = next(r for r in tables[0].regions if r.kind == "table")
        c = tables[0]
        rows = c.text[region.start - c.start : region.end - c.start].splitlines()
        reference = next(row for row in rows if "Reference" in row)
        candidate = next(row for row in rows if "Candidate" in row)
        assert reference.index("91.7") == candidate.index("95.4")
        other, _ = await frozen.search(["Separate results"], [doc.id], 1, "lexical")
        assert await frozen.read_tables(other, [doc.id]) == []
        with pytest.raises(ProviderError):
            await frozen.read_tables([seeds[0].model_copy(update={"page": 2})], [doc.id])
    finally:
        await container.close()


@pytest.mark.parametrize(
    ("source", "verdict"),
    [
        ("Method A scored 90% and method B scored 88%.", "insufficient"),
        ("A's gain was not statistically significant (p=0.72).", "insufficient"),
        ("The improvement of A over B was statistically significant.", "supported"),
    ],
)
def test_claim_scoped_statistical_evidence(evidence, source, verdict):
    ev = evidence.model_copy(update={"quote": source})
    unrelated = evidence.model_copy(update={"id": "other", "quote": "Statistically significant."})
    claim = Claim(id="c", text="Mức tăng của A so với B có ý nghĩa thống kê.", evidence_ids=[ev.id])
    check = ClaimCheck(claim_id="c", verdict="supported", evidence_ids=[ev.id], explanation="OK")
    assert validate_checks([claim], [ev, unrelated], [check])[0].verdict == verdict


def test_gold_passage_must_occur_on_named_page(tmp_path):
    (tmp_path / "study.pdf").write_bytes(table_pdf())
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "x",
                "question": "Which group was measured?",
                "files": ["study.pdf"],
                "gold_aspects": [
                    {
                        "text": "The other page",
                        "source": "study.pdf",
                        "passage": "Separate results on a different physical page.",
                        "page": 1,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="Gold passage"):
        load_cases(path)


def test_search_then_import_selected_arxiv_version(settings):
    feed = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
    <id>http://arxiv.org/abs/1706.03762v7</id><title>Attention Is All You Need</title>
    <author><name>A. Author</name></author><published>2023-01-01</published>
    </entry></feed>"""
    data = table_pdf()

    def upstream(request):
        if request.url.host == "export.arxiv.org":
            return httpx.Response(200, text=feed)
        assert str(request.url) == "https://arxiv.org/pdf/1706.03762v7"
        return httpx.Response(200, content=data)

    with TestClient(create_app(settings)) as api:
        with _mock_library(api, upstream):
            result = api.post("/api/papers/search", json={"query": "attention"})
            paper = result.json()[0]
            selected = {"paper_id": paper["paper_id"]}
            assert api.get("/api/documents").json()["documents"] == []
            assert (
                api.post("/api/papers/import", json={"paper_id": "../private"}).status_code == 400
            )
            doc = api.post("/api/papers/import", json=selected).json()["document"]
            assert doc["arxiv_id"] == "1706.03762v7" and doc["title"] == paper["title"]
            assert api.get(f"/api/documents/{doc['id']}/file").content == data
            assert api.post("/api/papers/import", json=selected).json()["duplicate"]
            assert len(api.get("/api/documents").json()["documents"]) == 1


@contextmanager
def _mock_library(api, handler):
    external = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    api.app.state.arxiv = ArxivLibrary(external)
    try:
        yield
    finally:
        api.portal.call(external.aclose)


async def test_arxiv_rejects_oversize_pdf_and_external_redirect():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"%PDF-" + b"x" * 100))
    ) as client:
        with pytest.raises(DocumentError):
            await ArxivLibrary(client).pdf("1706.03762v1", 16)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(302, headers={"Location": "http://127.0.0.1/private"})
        )
    ) as client:
        with pytest.raises(ProviderError):
            await ArxivLibrary(client).pdf("1706.03762v1", 128)
