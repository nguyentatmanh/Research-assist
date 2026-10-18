"""Dependency boundaries. An agent never imports a vendor SDK or SQL."""

from collections.abc import Callable
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from research_assist.domain.models import RetrievedChunk

T = TypeVar("T", bound=BaseModel)


class LanguageModel(Protocol):
    name: str
    model: str

    async def generate(
        self,
        *,
        task: str,
        instruction: str,
        payload: dict[str, object],
        schema: type[T],
        fallback: Callable[[], T],
    ) -> T: ...


class Embedder(Protocol):
    fingerprint: str | None

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]: ...


class SearchTool(Protocol):
    async def search(
        self,
        queries: list[str],
        document_ids: list[str],
        top_k: int,
        mode: str,
    ) -> tuple[list[RetrievedChunk], list[str]]: ...


class Reranker(Protocol):
    async def rank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


@runtime_checkable
class DiagnosticReranker(Reranker, Protocol):
    """Optional runtime diagnostics; not every reranker has API counters."""

    def diagnostics(self) -> dict[str, object]: ...
