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
from research_assist.infrastructure.cohere_reranker import CohereReranker
from research_assist.infrastructure.embeddings import (
    APIEmbeddings,
    LocalEmbeddings,
    NoEmbeddings,
    RemoteEmbeddings,
)
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
    search: HybridSearch

    async def close(self) -> None:
        await self.runs.close()
        await self.client.aclose()

    def configure_embeddings(self, settings: Settings) -> None:
        """Caller must ensure no run or document operation is active; no await during swap."""
        embedder = build_embedder(settings, self.client)
        search = HybridSearch(self.repository, embedder, settings)
        self.settings, self.embedder, self.search = settings, embedder, search
        self.documents.settings, self.documents.embedder = settings, embedder
        self.runs.settings = settings
        self.runs.pipeline.retrieval.search = search
        self.runs.pipeline.embedding_model = embedder.fingerprint


def build_embedder(settings: Settings, client: httpx.AsyncClient) -> Embedder:
    if settings.embedding_provider == "sentence-transformers":
        return LocalEmbeddings(
            settings.embedding_model,
            settings.embedding_batch_size,
            family=settings.embedding_family,
            device=settings.local_device,
            revision=settings.embedding_revision,
            max_seq_length=settings.embedding_max_seq_length,
        )
    if settings.embedding_provider == "openai":
        return APIEmbeddings(settings, client)
    if settings.embedding_provider == "remote":
        return RemoteEmbeddings(settings, client)
    return NoEmbeddings()


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
    embedder = build_embedder(settings, client)
    reranker: Reranker
    if settings.resolved_reranker_provider == "cohere":
        reranker = CohereReranker(settings, client, repository)
    elif settings.resolved_reranker_provider == "local":
        reranker = LocalReranker(
            settings.reranker_model,
            device=settings.local_device
            if settings.reranker_device == "inherit"
            else settings.reranker_device,
            batch_size=settings.reranker_batch_size,
            max_length=settings.reranker_max_length,
        )
    else:
        reranker = NoReranker()
    search = HybridSearch(repository, embedder, settings)
    pipeline = ResearchPipeline(
        llm,
        RetrievalAgent(
            llm,
            search,
            reranker,
            settings.rerank_candidates,
            settings.reranker_failure_policy == "fallback",
        ),
        ReaderAgent(llm),
        SummarizerAgent(llm),
        FactCheckerAgent(llm),
        WriterAgent(llm),
        embedder.fingerprint,
        settings.workflow_profile,
        settings.orchestration_mode,
        settings.coordinator_max_steps,
        settings.agent_max_steps,
        settings.autonomous_max_steps,
        settings.autonomous_max_tasks,
        settings.autonomous_parallelism,
        settings.autonomous_team_planning,
        settings.parallel_task_retries,
        settings.parallel_task_max_calls,
        settings.parallel_task_max_tokens,
    )
    return Container(
        settings,
        repository,
        DocumentService(repository, embedder, settings),
        RunManager(repository, pipeline, settings),
        client,
        embedder,
        llm,
        search,
    )
