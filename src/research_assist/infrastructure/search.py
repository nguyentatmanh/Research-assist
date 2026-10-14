"""BM25 + cosine, fused by reciprocal rank; score is ranking, not confidence."""

import asyncio
import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from research_assist.config import Settings
from research_assist.domain.errors import ProviderError
from research_assist.domain.models import IndexedChunk, RetrievedChunk
from research_assist.domain.ports import Embedder
from research_assist.domain.text import tokens
from research_assist.infrastructure.repository import Repository


def bm25_scores(query: str, corpus: list[str]) -> list[float]:
    return BM25Index(corpus).score(query)


class BM25Index:
    """Inverted index: repeated queries never tokenize the corpus again."""

    def __init__(self, corpus: list[str]) -> None:
        self.count = len(corpus)
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, text in enumerate(corpus):
            counts = Counter(tokens(text))
            self.lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((i, tf))
        self.average = max(sum(self.lengths) / max(self.count, 1), 1)

    def score(self, query: str) -> list[float]:
        scores = [0.0] * self.count
        for term in set(tokens(query)):
            postings = self.postings.get(term, [])
            df = len(postings)
            if not df:
                continue
            idf = math.log(1 + (self.count - df + 0.5) / (df + 0.5))
            for index, tf in postings:
                scores[index] += (
                    idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths[index] / self.average))
                )
        return scores


def chunk_view(chunk: IndexedChunk, **scores: Any) -> RetrievedChunk:
    return RetrievedChunk(**chunk.model_dump(exclude={"vector", "vector_model"}), **scores)


class HybridSearch:
    def __init__(self, repository: Repository, embedder: Embedder, settings: Settings) -> None:
        self.repository, self.embedder, self.settings = repository, embedder, settings

    async def search(
        self,
        queries: list[str],
        document_ids: list[str],
        top_k: int,
        mode: str,
    ) -> tuple[list[RetrievedChunk], list[str]]:
        chunks = await asyncio.to_thread(self.repository.chunks, document_ids)
        if not chunks:
            return [], []
        warnings: list[str] = []
        use_dense = mode == "hybrid" and self.embedder.fingerprint is not None
        if mode == "hybrid" and not use_dense:
            warnings.append(
                "Chưa cấu hình embedding: phiên này sử dụng BM25, không có tìm kiếm ngữ nghĩa."
            )
        dense_chunks = [
            c for c in chunks if c.vector_model == self.embedder.fingerprint and c.vector
        ]
        if use_dense and len(dense_chunks) != len(chunks):
            warnings.append(
                "Một số tài liệu chưa có embedding đúng model. "
                "Reindex để tìm kiếm ngữ nghĩa đầy đủ."
            )
        query_vectors = (
            await self.embedder.embed(queries, query=True) if use_dense and dense_chunks else []
        )
        return await asyncio.to_thread(
            self._rank, queries, chunks, dense_chunks, query_vectors, top_k, warnings
        )

    def _rank(
        self,
        queries: list[str],
        chunks: list[IndexedChunk],
        dense_chunks: list[IndexedChunk],
        query_vectors: list[list[float]],
        top_k: int,
        warnings: list[str],
    ) -> tuple[list[RetrievedChunk], list[str]]:
        corpus = [c.text for c in chunks]
        lexical_index = BM25Index(corpus)
        fused: dict[str, float] = defaultdict(float)
        lexical: dict[str, float] = defaultdict(float)
        semantic: dict[str, float] = {}
        for qindex, query in enumerate(queries):
            scores = lexical_index.score(query)
            indices = sorted(range(len(scores)), key=lambda i: (-scores[i], chunks[i].id))
            positive = [i for i in indices if scores[i] > 0][: max(40, top_k * 4)]
            for rank, index in enumerate(positive, 1):
                key = chunks[index].id
                fused[key] += 1 / (60 + rank)
                lexical[key] = max(lexical[key], scores[index])
            if query_vectors:
                query_vector = query_vectors[qindex]
                if any(len(c.vector or []) != len(query_vector) for c in dense_chunks):
                    raise ProviderError(
                        "Chiều embedding thay đổi. Reindex với cùng model trước khi nghiên cứu."
                    )
                matrix = np.asarray([c.vector for c in dense_chunks], dtype=np.float64)
                cosines = matrix @ np.asarray(query_vector, dtype=np.float64)
                dense_order = sorted(
                    range(len(cosines)), key=lambda i: (-cosines[i], dense_chunks[i].id)
                )
                dense_order = [
                    i for i in dense_order if cosines[i] >= self.settings.semantic_min_score
                ]
                for rank, index in enumerate(dense_order[: max(40, top_k * 4)], 1):
                    key = dense_chunks[index].id
                    fused[key] += 1 / (60 + rank)
                    semantic[key] = max(semantic.get(key, -1), float(cosines[index]))
        lookup = {c.id: c for c in chunks}
        ordered = sorted(fused, key=lambda key: (-fused[key], key))
        # Suppress almost-identical overlapping chunks from the same page.
        selected: list[RetrievedChunk] = []
        for key in ordered:
            chunk = lookup[key]
            if any(
                previous.document_id == chunk.document_id
                and previous.page == chunk.page
                and len(set(tokens(previous.text)) & set(tokens(chunk.text)))
                / max(1, len(set(tokens(previous.text)) | set(tokens(chunk.text))))
                > 0.92
                for previous in selected
            ):
                continue
            selected.append(
                chunk_view(
                    chunk,
                    score=fused[key],
                    lexical_score=lexical[key],
                    semantic_score=semantic.get(key),
                )
            )
            if len(selected) >= top_k:
                break
        return selected, warnings


class NoReranker:
    async def rank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks


class LocalReranker:
    def __init__(self, model: str, *, device: str = "cpu") -> None:
        self.model_name = model
        self.device = device
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def rank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._rank, query, chunks)

    def _rank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                options: dict[str, Any] = {"trust_remote_code": False}
                if self.device != "auto":
                    options["device"] = self.device
                self._model = CrossEncoder(self.model_name, **options)
            scores = self._model.predict([(query, c.text) for c in chunks]).tolist()
            if len(scores) != len(chunks) or any(not math.isfinite(float(s)) for s in scores):
                raise ValueError("invalid scores")
            items = [
                c.model_copy(update={"rerank_score": float(score)})
                for c, score in zip(chunks, scores, strict=True)
            ]
            return sorted(items, key=lambda c: -(c.rerank_score or 0))
        except Exception as exc:
            raise ProviderError(
                "Không chạy được reranker. Kiểm tra model hoặc tắt RA_RERANKER_MODEL."
            ) from exc
