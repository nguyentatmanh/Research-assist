from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from research_assist.api.app import create_app
from research_assist.config import Settings
from research_assist.infrastructure.arxiv import Paper


@pytest.mark.parametrize("unavailable", ["arxiv", "crossref"])
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
        getattr(app.state, unavailable).search = AsyncMock(side_effect=RuntimeError("unavailable"))
        getattr(app.state, available).search = AsyncMock(return_value=[paper])
        response = client.post("/api/scout/search", json={"query": "attention evidence"})
        assert response.status_code == 200
        result = response.json()
        assert len(result["papers"]) == 1
        assert result["papers"][0]["provider"] == available
        assert result["papers"][0]["year"] == 2017
        assert len(result["warnings"]) == 1
        assert result["warnings"][0].startswith(unavailable + ":")
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
