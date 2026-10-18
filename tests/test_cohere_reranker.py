import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from research_assist.agents.retrieval import RetrievalAgent
from research_assist.api.app import create_app
from research_assist.config import Settings
from research_assist.domain.errors import ProviderError
from research_assist.domain.models import ResearchRequest
from research_assist.infrastructure.cohere_reranker import CohereReranker
from research_assist.infrastructure.llm import DemoLanguageModel, usage_scope
from research_assist.infrastructure.repository import Repository


@pytest.fixture
def cohere_settings(tmp_path):
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        reranker_provider="cohere",
        cohere_api_key="secret-test-key",
        reranker_model="rerank-v4.0-pro",
    )


async def test_batch_cache_and_provenance(cohere_settings, chunk):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.1},
                ]
            },
        )

    other = chunk.model_copy(update={"id": "ch_other", "text": "Another exact source text."})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        ranker = CohereReranker(cohere_settings, client, Repository(cohere_settings.data_dir))
        ranked = await ranker.rank("Research question", [chunk, other])
        repeated = await ranker.rank("Research question", [chunk, other])
        assert len(requests) == 1
        assert [c.id for c in ranked] == [other.id, chunk.id]
        assert repeated == ranked
        assert ranked[0].text == other.text and ranked[0].start == other.start
        assert chunk.rerank_score is None  # Original indexed view is immutable.
        assert ranker.diagnostics()["cache_hits"] == 1
        payload = json.loads(requests[0].content)
        assert payload["documents"] == [chunk.text, other.text]
        assert payload["top_n"] == 2
        assert str(requests[0].url) == "https://api.cohere.com/v2/rerank"
        assert requests[0].headers["Authorization"] == "Bearer secret-test-key"
        changed = other.model_copy(update={"text": "Updated source contents."})
        await ranker.rank("Research question", [chunk, changed])
        assert len(requests) == 2  # Cache cannot survive changed evidence text.


@pytest.mark.parametrize(
    "results",
    [
        [],
        [{"index": 0, "relevance_score": 0.1}, {"index": 0, "relevance_score": 0.9}],
        [{"index": 2, "relevance_score": 0.9}],
        [{"index": True, "relevance_score": 0.9}],
        [{"index": "0", "relevance_score": 0.9}],
        [{"index": 0, "relevance_score": "0.9"}],
        [{"index": 0, "relevance_score": 1.2}],
        [{"index": 0, "relevance_score": -0.1}],
    ],
)
async def test_invalid_results_fail_closed(cohere_settings, chunk, results):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"results": results})
        )
    ) as client:
        ranker = CohereReranker(cohere_settings, client, Repository(cohere_settings.data_dir))
        with pytest.raises(ProviderError, match="không hợp lệ"):
            await ranker.rank("Question", [chunk])
        assert ranker.failures == 1


@pytest.mark.parametrize("status", [401, 403, 429, 498, 500])
async def test_errors_are_safe_and_never_automatically_retried(cohere_settings, chunk, status):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            status, text="secret-test-key PRIVATE_DOCUMENT", headers={"Retry-After": "60"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        ranker = CohereReranker(cohere_settings, client, Repository(cohere_settings.data_dir))
        with pytest.raises(ProviderError) as exc:
            await ranker.rank("Question", [chunk])
        assert "secret-test-key" not in str(exc.value)
        assert "PRIVATE_DOCUMENT" not in str(exc.value)
        assert len(requests) == 1
        assert ranker.last_rank["failed"] is True
        if status == 429:
            assert ranker._blocked_until > time.monotonic() + 50
            with pytest.raises(ProviderError, match="tạm ngưng"):
                await ranker.rank("Another question", [chunk])
            assert len(requests) == 1


async def test_concurrent_identical_calls_share_one_request(cohere_settings, chunk):
    requests = []

    async def respond(request):
        requests.append(request)
        await asyncio.sleep(0)
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.8}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        ranker = CohereReranker(cohere_settings, client, Repository(cohere_settings.data_dir))
        await asyncio.gather(ranker.rank("Question", [chunk]), ranker.rank("Question", [chunk]))
        assert len(requests) == 1 and ranker.cache_hits == 1


async def test_cache_opt_out_and_empty_input(cohere_settings, chunk):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.8}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        ranker = CohereReranker(cohere_settings, client, Repository(cohere_settings.data_dir))
        assert await ranker.rank("Question", []) == []
        with usage_scope(cache_enabled=False):
            await ranker.rank("Question", [chunk])
            await ranker.rank("Question", [chunk])
        assert len(requests) == 2 and ranker.cache_hits == 0


async def test_rate_gate_waits_before_sending_when_budget_is_full(cohere_settings, chunk):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.8}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        ranker = CohereReranker(cohere_settings, client, Repository(cohere_settings.data_dir))
        ranker.gate.timestamps.extend([time.monotonic()] * ranker.gate.rpm)
        task = asyncio.create_task(ranker.rank("Question", [chunk]))
        await asyncio.sleep(0.02)
        assert not requests and not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_config_and_diagnostics_do_not_disclose_key(cohere_settings):
    with TestClient(create_app(cohere_settings)) as client:
        config = client.get("/api/config").json()
        diagnostics = client.get("/api/diagnostics").json()
        assert config["reranker_provider"] == "cohere" and not config["demo"]
        assert diagnostics["reranker"]["api"]["api_calls"] == 0
        assert "secret-test-key" not in json.dumps([config, diagnostics])


def test_legacy_and_missing_key_configuration():
    assert Settings(_env_file=None).resolved_reranker_provider == "none"
    assert (
        Settings(_env_file=None, reranker_model="local-model").resolved_reranker_provider == "local"
    )
    assert (
        Settings(
            _env_file=None, reranker_provider="none", reranker_model="x"
        ).resolved_reranker_model
        == ""
    )
    with pytest.raises(ValidationError, match="RA_COHERE_API_KEY"):
        Settings(_env_file=None, reranker_provider="cohere")


async def test_timeout_is_a_safe_provider_error(cohere_settings, chunk):
    def respond(request):
        raise httpx.ReadTimeout("secret-test-key", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        ranker = CohereReranker(cohere_settings, client, Repository(cohere_settings.data_dir))
        with pytest.raises(ProviderError, match="hết thời gian") as exc:
            await ranker.rank("Question", [chunk])
        assert "secret-test-key" not in str(exc.value)


@pytest.mark.parametrize("fallback", [True, False])
async def test_retrieval_fallback_is_explicit_in_trace(cohere_settings, chunk, fallback):
    class Search:
        async def search(self, queries, document_ids, top_k, mode):
            return [chunk], []

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(429))
    ) as client:
        ranker = CohereReranker(cohere_settings, client, Repository(cohere_settings.data_dir))
        agent = RetrievalAgent(DemoLanguageModel(), Search(), ranker, reranker_fallback=fallback)
        request = ResearchRequest(question="Explain source evidence")
        if not fallback:
            with pytest.raises(ProviderError, match="hạn mức"):
                await agent.run(request, [])
        else:
            plan, chunks, warnings = await agent.run(request, [])
            assert chunks == [chunk] and any("hạn mức" in w for w in warnings)
            assert plan.trace[0]["reranker"]["provider"] == "cohere"
            assert plan.trace[0]["reranker"]["fallback"] is True
            assert plan.trace[0]["reranker"]["api_calls"] == 1
