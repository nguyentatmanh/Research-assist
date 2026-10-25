from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from research_assist.config import Settings
from research_assist.domain.models import Chunk, Document, Evidence, RetrievedChunk
from research_assist.domain.ports import T


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # Existing fixtures exercise the fixed baseline; adaptive tests opt in explicitly.
    return Settings(data_dir=tmp_path, orchestration_mode="fixed", _env_file=None)


@pytest.fixture
def document() -> Document:
    return Document(
        id="doc_test",
        filename="study.pdf",
        title="Study",
        kind="pdf",
        sha256="abc",
        page_count=2,
        chunk_count=1,
        created_at="2026-10-09T00:00:00+00:00",
    )


@pytest.fixture
def chunk(document: Document) -> RetrievedChunk:
    text = "RAG combines retrieval with generation. The experiment used 50 documents, not 500."
    return RetrievedChunk(
        id="ch_test",
        document_id=document.id,
        document_title=document.title,
        filename=document.filename,
        text=text,
        page=2,
        section="Trang 2",
        ordinal=0,
        start=10,
        end=10 + len(text),
        score=0.02,
    )


@pytest.fixture
def evidence(chunk: Chunk) -> Evidence:
    quote = "RAG combines retrieval with generation."
    return Evidence(
        id="ev_test",
        chunk_id=chunk.id,
        quote=quote,
        relevance="Relevant",
        document_id=chunk.document_id,
        document_title=chunk.document_title,
        filename=chunk.filename,
        page=chunk.page,
        section=chunk.section,
        start=chunk.start,
        end=chunk.start + len(quote),
    )


class ScriptedLLM:
    name: str = "test"
    model: str = "test-model"

    def __init__(self, outputs: dict[str, dict[str, Any]]) -> None:
        self.outputs = outputs
        self.calls: list[str] = []
        self.payloads: list[dict[str, object]] = []

    async def generate(
        self,
        *,
        task: str,
        instruction: str,
        payload: dict[str, object],
        schema: type[T],
        fallback: Callable[[], T],
    ) -> T:
        self.calls.append(task)
        self.payloads.append(payload)
        return schema.model_validate(self.outputs[task])
