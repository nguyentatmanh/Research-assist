import sys
from types import SimpleNamespace

import numpy as np
import pytest

from research_assist.domain.models import RetrievedChunk
from research_assist.infrastructure.embeddings import LocalEmbeddings
from research_assist.infrastructure.search import LocalReranker


async def test_e5_adapter_reuses_model_and_applies_query_passage_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = []
    encoded = []

    class Encoder:
        def __init__(self, model: str, **kwargs: object) -> None:
            loaded.append((model, kwargs))

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            encoded.append(texts)
            return np.asarray([[3.0, 4.0] for _ in texts])

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Encoder)
    )
    embedder = LocalEmbeddings("intfloat/multilingual-e5-small", 16)
    passages = await embedder.embed(["tài liệu nghiên cứu"])
    queries = await embedder.embed(["câu hỏi nghiên cứu"], query=True)
    assert passages == queries == [[0.6, 0.8]]
    assert len(loaded) == 1 and loaded[0][1]["trust_remote_code"] is False
    assert encoded == [["passage: tài liệu nghiên cứu"], ["query: câu hỏi nghiên cứu"]]


async def test_reranker_orders_by_cross_encoder_score(
    chunk: RetrievedChunk, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Ranker:
        def __init__(self, _: str, **kwargs: object) -> None:
            assert kwargs["trust_remote_code"] is False

        def predict(
            self,
            pairs: list[tuple[str, str]],
            *,
            batch_size: int,
            show_progress_bar: bool,
        ) -> np.ndarray:
            assert pairs == [("RAG question", chunk.text), ("RAG question", chunk.text)]
            assert batch_size == 4
            assert show_progress_bar is False
            return np.asarray([0.1, 0.8])

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=Ranker))
    other = chunk.model_copy(update={"id": "ch_other"})
    ordered = await LocalReranker("test-reranker").rank("RAG question", [chunk, other])
    assert ordered[0].id == "ch_other" and ordered[0].rerank_score == 0.8
