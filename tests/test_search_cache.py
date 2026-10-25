import asyncio

import pytest

from research_assist.agents.retrieval import RetrievalAgent
from research_assist.application.documents import DocumentService
from research_assist.config import Settings
from research_assist.domain.errors import ProviderError
from research_assist.domain.models import ResearchRequest
from research_assist.infrastructure.embeddings import NoEmbeddings
from research_assist.infrastructure.llm import DemoLanguageModel
from research_assist.infrastructure.repository import Repository
from research_assist.infrastructure.search import HybridSearch
from tests.test_retrieval import TinyEmbedder


async def test_query_prefetch_batches_unique_vectors_and_respects_model_identity(settings):
    class CountingEmbedder(TinyEmbedder):
        def __init__(self):
            self.queries = []

        async def embed(self, texts, *, query=False):
            if query:
                self.queries.append(list(texts))
            return await super().embed(texts, query=query)

    repo, embedder = Repository(settings.data_dir), CountingEmbedder()
    doc, _ = await DocumentService(repo, embedder, settings).ingest(
        b"Distant galaxies contain billions of stars and planetary systems.", "science.txt"
    )
    search = HybridSearch(repo, embedder, settings)
    await search.prepare_queries(["astronomy", "stars", "astronomy"], [doc.id], "hybrid")
    for query in ("astronomy", "stars"):
        assert (await search.search([query], [doc.id], 4, "hybrid"))[0]
    assert embedder.queries == [["astronomy", "stars"]]
    embedder.fingerprint = "test:new-space"
    await search._query_vectors(["astronomy"])
    assert len(embedder.queries) == 2
    assert search.diagnostics()["query_cache_entries"] == 3


async def test_snapshot_reused_then_invalidated_for_other_repository_writes(
    settings: Settings,
) -> None:
    repository = Repository(settings.data_dir)
    service = DocumentService(repository, NoEmbeddings(), settings)
    doc, _ = await service.ingest(
        b"RAG uses retrieval and generation with source evidence.", "one.txt"
    )
    search = HybridSearch(repository, NoEmbeddings(), settings)
    results = await asyncio.gather(*(search.search(["RAG"], [], 4, "lexical") for _ in range(3)))
    assert all(r[0][0].document_id == doc.id for r in results)
    assert search.diagnostics()["builds"] == 1 and search.diagnostics()["hits"] == 2
    other_repo = Repository(settings.data_dir)
    await DocumentService(other_repo, NoEmbeddings(), settings).ingest(
        b"RAG also uses document retrieval to support an answer with citations.", "two.txt"
    )
    found, _ = await search.search(["RAG"], [], 4, "lexical")
    assert len(found) == 2 and search.diagnostics()["builds"] == 2
    other_repo.delete_document(doc.id)
    found, _ = await search.search(["RAG"], [], 4, "lexical")
    assert all(c.document_id != doc.id for c in found) and search.diagnostics()["builds"] == 3


async def test_reindex_invalidates_cached_vectors(settings: Settings) -> None:
    repo, embedder = Repository(settings.data_dir), TinyEmbedder()
    service = DocumentService(repo, embedder, settings)
    await service.ingest(
        b"Distant galaxies contain billions of stars and planetary systems.", "science.txt"
    )
    search = HybridSearch(repo, embedder, settings)
    assert (await search.search(["astronomy"], [], 4, "hybrid"))[0]
    chunks = repo.chunks()
    chunks[0].vector = [0.0, 1.0]
    repo.update_vectors(chunks)
    assert not (await search.search(["astronomy"], [], 4, "hybrid"))[0]
    assert search.diagnostics()["builds"] == 2


async def test_scope_cache_cannot_leak_and_lru_is_bounded(settings: Settings) -> None:
    settings.search_cache_scopes = 1
    repo = Repository(settings.data_dir)
    service = DocumentService(repo, NoEmbeddings(), settings)
    a, _ = await service.ingest(b"RAG uses source evidence for generation and retrieval.", "a.txt")
    b, _ = await service.ingest(
        b"RAG source evidence should remain private to the selected scope.", "b.txt"
    )
    search = HybridSearch(repo, NoEmbeddings(), settings)
    await search.search(["RAG"], [a.id], 4, "lexical")
    rows, _ = await search.search(["RAG"], [b.id], 4, "lexical")
    assert {c.document_id for c in rows} == {b.id}
    assert search.diagnostics()["scopes"] == 1


async def test_reranker_failure_returns_retrieval_order_with_warning(settings: Settings) -> None:
    class BrokenReranker:
        async def rank(self, query, chunks):
            raise ProviderError("test-only outage")

    repo = Repository(settings.data_dir)
    await DocumentService(repo, NoEmbeddings(), settings).ingest(
        b"RAG uses source evidence for generation and retrieval.", "source.txt"
    )
    search = HybridSearch(repo, NoEmbeddings(), settings)
    agent = RetrievalAgent(DemoLanguageModel(), search, BrokenReranker())
    _, chunks, warnings = await agent.run(
        ResearchRequest(question="Explain RAG source evidence"), []
    )
    assert chunks and any("Reranker" in x for x in warnings)
    agent.reranker_fallback = False
    with pytest.raises(ProviderError):
        await agent.run(ResearchRequest(question="Explain RAG source evidence"), [])


async def test_lexical_mode_remains_available_with_bad_dense_vectors(settings: Settings) -> None:
    repo, embedder = Repository(settings.data_dir), TinyEmbedder()
    service = DocumentService(repo, embedder, settings)
    await service.ingest(
        b"RAG retrieval uses galaxies and scientific source evidence.", "source.txt"
    )
    chunks = repo.chunks()
    chunks[0].vector = [0.0, 0.0]
    repo.update_vectors(chunks)
    search = HybridSearch(repo, embedder, settings)
    assert (await search.search(["RAG"], [], 4, "lexical"))[0]
    with pytest.raises(ProviderError):
        await search.search(["RAG"], [], 4, "hybrid")
