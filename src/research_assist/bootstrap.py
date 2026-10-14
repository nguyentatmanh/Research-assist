"""Composition root: the only place concrete infrastructure is wired into agents."""

from dataclasses import dataclass

import httpx

from research_assist.agents.fact_checker import FactCheckerAgent
from research_assist.agents.reader import ReaderAgent
from research_assist.agents.retrieval import RetrievalAgent
from research_assist.agents.summarizer import SummarizerAgent
from research_assist.agents.writer import WriterAgent
from research_assist.application.documents import DocumentService
from research_assist.application.pipeline import ResearchPipeline
from research_assist.application.runs import RunManager
from research_assist.config import Settings
from research_assist.domain.ports import Embedder, Reranker
from research_assist.infrastructure.embeddings import APIEmbeddings, LocalEmbeddings, NoEmbeddings
from research_assist.infrastructure.repository import Repository
from research_assist.infrastructure.routing import ModelRouter
from research_assist.infrastructure.search import HybridSearch, LocalReranker, NoReranker


@dataclass
class Container:
    settings: Settings
    repository: Repository
    documents: DocumentService
    runs: RunManager
    client: httpx.AsyncClient
    embedder: Embedder
    router: ModelRouter

    async def close(self) -> None:
        await self.runs.close()
        await self.client.aclose()


def build_container(
    settings: Settings, transport: httpx.AsyncBaseTransport | None = None
) -> Container:
    repository = Repository(settings.data_dir)
    repository.recover_interrupted()
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=10),
        limits=httpx.Limits(max_connections=12, max_keepalive_connections=8),
        follow_redirects=False,
        transport=transport,
    )
    llm = ModelRouter(settings, client, repository)
    embedder: Embedder = NoEmbeddings()
    if settings.embedding_provider == "sentence-transformers":
        embedder = LocalEmbeddings(
            settings.embedding_model,
            settings.embedding_batch_size,
            family=settings.embedding_family,
            device=settings.local_device,
            revision=settings.embedding_revision,
            max_seq_length=settings.embedding_max_seq_length,
        )
    elif settings.embedding_provider == "openai":
        embedder = APIEmbeddings(settings, client)
    reranker: Reranker = (
        LocalReranker(settings.reranker_model, device=settings.local_device)
        if settings.reranker_model
        else NoReranker()
    )
    search = HybridSearch(repository, embedder, settings)
    pipeline = ResearchPipeline(
        llm,
        RetrievalAgent(llm, search, reranker, settings.rerank_candidates),
        ReaderAgent(llm),
        SummarizerAgent(llm),
        FactCheckerAgent(llm),
        WriterAgent(llm),
        embedder.fingerprint,
        settings.workflow_profile,
    )
    return Container(
        settings,
        repository,
        DocumentService(repository, embedder, settings),
        RunManager(repository, pipeline, settings),
        client,
        embedder,
        llm,
    )
