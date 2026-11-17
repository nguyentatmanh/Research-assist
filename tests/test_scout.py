import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from research_assist.api.app import create_app
from research_assist.application import literature_scout
from research_assist.application.literature_scout import ScoutCache
from research_assist.config import Settings
from research_assist.domain.errors import CapacityError
from research_assist.infrastructure.arxiv import Paper


@pytest.mark.parametrize("unavailable", ["arxiv", "crossref", "openalex"])
def test_scout_keeps_results_when_one_provider_fails(settings: Settings, unavailable: str) -> None:
    app = create_app(settings)
    available = "crossref" if unavailable == "arxiv" else "arxiv"
    paper = Paper(
        paper_id="1706.03762v7",
        title="Attention Is All You Need",
        authors=["Researcher"],
        published="2017-06-12",
        url="https://arxiv.org/abs/1706.03762v7",
    )
    with TestClient(app) as client:
        for provider in ("arxiv", "crossref", "openalex"):
            getattr(app.state, provider).search = AsyncMock(return_value=[])
        getattr(app.state, unavailable).search = AsyncMock(side_effect=CapacityError("quota"))
        getattr(app.state, available).search = AsyncMock(return_value=[paper])
        response = client.post("/api/scout/search", json={"query": "attention evidence"})
        assert response.status_code == 200
        result = response.json()
        assert len(result["papers"]) == 1
        assert result["papers"][0]["provider"] == available
        assert result["papers"][0]["year"] == 2017
        assert len(result["warnings"]) == 1
        assert result["warnings"][0].casefold().startswith(unavailable + ":")
        assert client.get("/api/documents").json()["documents"] == []


def test_scout_deduplicates_titles_without_importing_metadata(settings: Settings) -> None:
    app = create_app(settings)
    paper = Paper(
        paper_id="1706.03762v7",
        title="Attention Is All You Need",
        authors=[],
        published="2017",
        url="https://arxiv.org/abs/1706.03762v7",
    )
    with TestClient(app) as client:
        app.state.arxiv.search = AsyncMock(return_value=[paper])
        app.state.openalex.search = AsyncMock(return_value=[])
        app.state.crossref.search = AsyncMock(
            return_value=[
                paper.model_copy(
                    update={"title": "  ATTENTION is all you need ", "source": "crossref"}
                ),
            ]
        )
        result = client.post("/api/scout/search", json={"query": "attention evidence"}).json()
        assert len(result["papers"]) == 1
        assert result["papers"][0]["arxiv_id"] == paper.paper_id
        assert result["warnings"] == []
        assert client.get("/api/documents").json()["documents"] == []


def test_scout_merges_doi_preserves_import_and_keeps_distinct_works(settings: Settings) -> None:
    app = create_app(settings)
    preprint = Paper(
        paper_id="1706.03762v7",
        title="A study of attention",
        authors=["A. Researcher"],
        published="2017",
        url="https://arxiv.org/abs/1706.03762v7",
    )
    published = Paper(
        paper_id="doi:10.1234/attention",
        title=preprint.title,
        authors=preprint.authors,
        published="2017-06-12",
        url="https://doi.org/10.1234/attention",
        source="crossref",
        doi="https://DOI.org/10.1234/ATTENTION",
        import_available=False,
    )
    openalex = Paper(
        paper_id="W123456",
        title="The study of attention",
        authors=preprint.authors,
        published="2017-06-12",
        url="https://openalex.org/W123456",
        source="openalex",
        doi="10.1234/attention",
        import_available=False,
        abstract="An abstract reconstructed from OpenAlex.",
    )
    distinct = openalex.model_copy(
        update={"paper_id": "W654321", "doi": "10.1234/reply", "title": preprint.title}
    )
    with TestClient(app) as client:
        app.state.arxiv.search = AsyncMock(return_value=[preprint])
        app.state.crossref.search = AsyncMock(return_value=[published])
        app.state.openalex.search = AsyncMock(return_value=[openalex, distinct])

        response = client.post("/api/scout/search", json={"query": "attention evidence"})
        assert response.status_code == 200
        merged, separate = response.json()["papers"]
        assert merged["doi"] == "10.1234/attention"
        assert merged["providers"] == ["arxiv", "crossref", "openalex"]
        assert merged["abstract"] == openalex.abstract
        assert merged["import_available"] and merged["arxiv_id"] == preprint.paper_id
        assert separate["doi"] == "10.1234/reply" and not separate["import_available"]
        assert client.get("/api/documents").json()["documents"] == []

        # A DOI lookup must exclude keyword matches from other providers.
        app.state.openalex.search.return_value = [openalex]
        app.state.crossref.search.return_value = [published, distinct]
        response = client.post("/api/scout/search", json={"query": "doi:10.1234/attention"})
        assert response.status_code == 200
        (exact,) = response.json()["papers"]
        assert exact["doi"] == "10.1234/attention"
        assert exact["providers"] == ["crossref", "openalex"]

        response = client.post(
            "/api/papers/search", json={"source": "openalex", "query": "10.1234/attention"}
        )
        assert response.status_code == 200
        assert response.json()[0]["source"] == "openalex"
        assert not response.json()[0]["import_available"]
        app.state.openalex.search.assert_awaited_with("10.1234/attention")
        assert app.state.openalex.search.await_count == 2


@pytest.mark.parametrize("cache_enabled", [True, False])
def test_scout_reuses_results_in_both_routes_and_keeps_arxiv_import(
    settings: Settings, cache_enabled: bool
) -> None:
    app = create_app(settings.model_copy(update={"cache_enabled": cache_enabled}))
    paper = Paper(
        paper_id="1706.03762v7",
        title="Attention Is All You Need",
        authors=[],
        published="2017",
        url="https://arxiv.org/abs/1706.03762v7",
    )
    with TestClient(app) as client:
        app.state.arxiv.search = AsyncMock(return_value=[paper])
        app.state.crossref.search = AsyncMock(return_value=[])
        app.state.openalex.search = AsyncMock(return_value=[])
        query = {"query": "attention evidence"}
        first = client.post("/api/scout/search", json=query)
        assert first.status_code == 200
        app.state.arxiv.recent.clear()
        cached = client.post("/api/papers/search", json={"query": " attention  evidence "})
        assert cached.status_code == 200 and cached.json()[0]["paper_id"] == paper.paper_id
        assert paper.paper_id in app.state.arxiv.recent
        assert client.post("/api/scout/search", json=query).json() == first.json()
        assert app.state.arxiv.search.await_count == (1 if cache_enabled else 3)


async def test_scout_cache_ttl_lru_and_provider_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    monkeypatch.setattr(literature_scout, "monotonic", lambda: now[0])
    cache = ScoutCache(ttl_seconds=86400, max_entries=2)
    paper = Paper(paper_id="W123", title="Example", authors=[], published="2025", url="")
    fetch = AsyncMock(return_value=[paper])

    await cache.search("openalex", "keyword", fetch)
    await cache.search("openalex", "10.1234/sample", fetch)
    now[0] = 3601
    await cache.search("openalex", "https://doi.org/10.1234/SAMPLE", fetch)
    assert fetch.await_count == 2
    await cache.search("openalex", "keyword", fetch)
    await cache.search("crossref", "keyword", fetch)
    await cache.search("openalex", "doi:10.1234/sample", fetch)
    # Keyword expired after one hour; the DOI stayed cached until LRU eviction.
    assert fetch.await_count == 5
    await cache.close()


async def test_scout_cache_coalesces_and_keeps_shared_request_alive() -> None:
    cache = ScoutCache(ttl_seconds=86400)
    started, release = asyncio.Event(), asyncio.Event()
    paper = Paper(paper_id="W123", title="Example", authors=[], published="2025", url="")

    async def respond(query: str) -> list[Paper]:
        assert query == "10.1234/sample"
        started.set()
        await release.wait()
        return [paper]

    fetch = AsyncMock(side_effect=respond)
    first = asyncio.create_task(cache.search("openalex", "10.1234/sample", fetch))
    await asyncio.wait_for(started.wait(), timeout=2)
    second = asyncio.create_task(cache.search("openalex", "doi:10.1234/SAMPLE", fetch))
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    result = await asyncio.wait_for(second, timeout=2)
    result[0].title = "changed by the caller"
    assert (await cache.search("openalex", "10.1234/sample", fetch))[0].title == "Example"
    fetch.assert_awaited_once()
    await cache.close()


async def test_scout_cache_retries_transient_failures() -> None:
    cache = ScoutCache(ttl_seconds=86400)
    fetch = AsyncMock(side_effect=[CapacityError("quota"), []])
    with pytest.raises(CapacityError):
        await cache.search("openalex", "query", fetch)
    assert await cache.search("openalex", "query", fetch) == []
    assert await cache.search("openalex", "query", fetch) == []
    assert fetch.await_count == 2
    await cache.close()
