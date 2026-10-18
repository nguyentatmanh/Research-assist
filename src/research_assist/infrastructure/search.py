"""BM25 + cosine, fused by reciprocal rank; score is ranking, not confidence."""

import asyncio
import math
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from research_assist.config import Settings
from research_assist.domain.errors import ProviderError
from research_assist.domain.models import IndexedChunk, RetrievedChunk
from research_assist.domain.ports import Embedder
from research_assist.domain.text import tokens
from research_assist.infrastructure.embeddings import embedding_identity, validate_vectors
from research_assist.infrastructure.local_runtime import INFERENCE_LOCK
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


@dataclass
class SearchSnapshot:
    revision: int
    chunks: list[IndexedChunk]
    lexical: BM25Index
    dense_chunks: list[IndexedChunk]
    matrix: npt.NDArray[np.float32]
    terms: dict[str, set[str]]
    estimated_bytes: int


class HybridSearch:
    def __init__(self, repository: Repository, embedder: Embedder, settings: Settings) -> None:
        self.repository, self.embedder, self.settings = repository, embedder, settings
        self._cache: OrderedDict[tuple[tuple[str, ...], str | None], SearchSnapshot] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = self._builds = 0

    def diagnostics(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "builds": self._builds,
            "scopes": len(self._cache),
            "estimated_bytes": sum(s.estimated_bytes for s in self._cache.values()),
        }

    def _build(self, scope: list[str], fingerprint: str | None) -> SearchSnapshot:
        revision, chunks = self.repository.retrieval_snapshot(scope)
        identity = embedding_identity(fingerprint)
        dense = [
            c
            for c in chunks
            if identity and embedding_identity(c.vector_model) == identity and c.vector
        ]
        try:
            matrix = (
                np.asarray([c.vector for c in dense], dtype=np.float32)
                if dense
                else np.empty((0, 0), dtype=np.float32)
            )
            if matrix.ndim != 2 or not np.isfinite(matrix).all():
                raise ValueError("Invalid vector matrix")
            if dense:
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                if not np.isfinite(norms).all() or np.any(norms <= 0):
                    raise ValueError("Invalid vector norm")
                matrix = matrix / norms
            matrix.setflags(write=False)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ProviderError(
                "Vector không hợp lệ hoặc khác chiều. Reindex kho tài liệu."
            ) from exc
        # Keep one compact float32 matrix; release Python lists of vector floats.
        for chunk in chunks:
            chunk.vector = None
        lexical = BM25Index([c.text for c in chunks])
        terms = {c.id: set(tokens(c.text)) for c in chunks}
        estimate = (
            matrix.nbytes
            + sum(len(c.text.encode()) * 3 + 1200 for c in chunks)
            + sum(len(p) * 128 for p in lexical.postings.values())
            + sum(sum(len(t) * 4 + 80 for t in ts) for ts in terms.values())
        )
        return SearchSnapshot(revision, chunks, lexical, dense, matrix, terms, estimate)

    async def _snapshot(self, scope: list[str], use_dense: bool) -> SearchSnapshot:
        key = (tuple(sorted(set(scope))), self.embedder.fingerprint if use_dense else None)
        async with self._lock:
            revision = await asyncio.to_thread(self.repository.corpus_revision)
            self._cache = OrderedDict(
                (k, v) for k, v in self._cache.items() if v.revision == revision
            )
            cached = self._cache.get(key)
            if cached is not None:
                self._hits += 1
                self._cache.move_to_end(key)
                return cached
            snapshot = await asyncio.to_thread(self._build, list(key[0]), key[1])
            self._builds += 1
            limit = self.settings.search_cache_mb * 1024 * 1024
            if self.settings.search_cache_scopes and snapshot.estimated_bytes <= limit:
                self._cache[key] = snapshot
                while (
                    len(self._cache) > self.settings.search_cache_scopes
                    or sum(x.estimated_bytes for x in self._cache.values()) > limit
                ):
                    self._cache.popitem(last=False)
            return snapshot

    async def search(
        self, queries: list[str], document_ids: list[str], top_k: int, mode: str
    ) -> tuple[list[RetrievedChunk], list[str]]:
        use_dense = mode == "hybrid" and self.embedder.fingerprint is not None
        snapshot = await self._snapshot(document_ids, use_dense)
        if not snapshot.chunks or not queries or top_k <= 0:
            return [], []
        warnings: list[str] = []
        if mode == "hybrid" and not use_dense:
            warnings.append(
                "Chưa cấu hình embedding: phiên này sử dụng BM25, không có tìm kiếm ngữ nghĩa."
            )
        if use_dense and len(snapshot.dense_chunks) != len(snapshot.chunks):
            warnings.append(
                "Một số tài liệu chưa có embedding đúng model. "
                "Reindex để tìm kiếm ngữ nghĩa đầy đủ."
            )
        vectors = (
            await self.embedder.embed(queries, query=True)
            if use_dense and snapshot.dense_chunks
            else []
        )
        if vectors:
            vectors = validate_vectors(vectors, len(queries))
        elif use_dense and snapshot.dense_chunks:
            raise ProviderError("Embedding không trả về vector cho truy vấn.")
        return await asyncio.to_thread(self._rank, queries, snapshot, vectors, top_k, warnings)

    async def expand_context(
        self, selected: list[RetrievedChunk], document_ids: list[str]
    ) -> list[RetrievedChunk]:
        """Merge one overlapping neighbor on the same physical page, up to 2600 chars.

        Index vectors remain child vectors; only Reader context expands. Exact offsets survive.
        Legacy chunks and unrecognized headings also work; gaps are never fabricated.
        """
        snapshot = await self._snapshot(document_ids, False)
        by_page: dict[tuple[str, int | None], list[IndexedChunk]] = defaultdict(list)
        for indexed in snapshot.chunks:
            by_page[(indexed.document_id, indexed.page)].append(indexed)
        expanded = []
        for chunk in selected:
            start, end, text = chunk.start, chunk.end, chunk.text
            neighbors = sorted(
                by_page[(chunk.document_id, chunk.page)],
                key=lambda c: abs(c.ordinal - chunk.ordinal),
            )
            for neighbor in neighbors:
                if neighbor.id == chunk.id or abs(neighbor.ordinal - chunk.ordinal) != 1:
                    continue
                if neighbor.section != chunk.section:
                    continue
                left, right = max(start, neighbor.start), min(end, neighbor.end)
                if left > right or max(end, neighbor.end) - min(start, neighbor.start) > 2600:
                    continue
                if (
                    text[left - start : right - start]
                    != neighbor.text[left - neighbor.start : right - neighbor.start]
                ):
                    continue
                if neighbor.start < start:
                    text = neighbor.text[: start - neighbor.start] + text
                    start = neighbor.start
                if neighbor.end > end:
                    text += neighbor.text[end - neighbor.start :]
                    end = neighbor.end
                break
            expanded.append(chunk.model_copy(update={"text": text, "start": start, "end": end}))
        return expanded

    def _rank(
        self,
        queries: list[str],
        snapshot: SearchSnapshot,
        query_vectors: list[list[float]],
        top_k: int,
        warnings: list[str],
    ) -> tuple[list[RetrievedChunk], list[str]]:
        chunks, dense = snapshot.chunks, snapshot.dense_chunks
        fused: dict[str, float] = defaultdict(float)
        lexical: dict[str, float] = defaultdict(float)
        semantic: dict[str, float] = {}
        cap = max(40, top_k * 4)
        for qindex, query in enumerate(queries):
            scores = snapshot.lexical.score(query)
            positive = sorted(
                (i for i, score in enumerate(scores) if score > 0),
                key=lambda i: (-scores[i], chunks[i].id),
            )[:cap]
            for rank, index in enumerate(positive, 1):
                key = chunks[index].id
                fused[key] += 1 / (60 + rank)
                lexical[key] = max(lexical[key], scores[index])
            if query_vectors:
                if len(query_vectors[qindex]) != snapshot.matrix.shape[1]:
                    raise ProviderError(
                        "Chiều embedding thay đổi. Reindex với cùng model trước khi nghiên cứu."
                    )
                cosines = np.clip(
                    snapshot.matrix @ np.asarray(query_vectors[qindex], dtype=np.float32), -1, 1
                )
                order = sorted(
                    (
                        i
                        for i, score in enumerate(cosines)
                        if score >= self.settings.semantic_min_score
                    ),
                    key=lambda i: (-cosines[i], dense[i].id),
                )[:cap]
                for rank, index in enumerate(order, 1):
                    key = dense[index].id
                    fused[key] += 1 / (60 + rank)
                    semantic[key] = max(semantic.get(key, -1), float(cosines[index]))
        lookup = {c.id: c for c in chunks}
        selected: list[RetrievedChunk] = []
        for key in sorted(fused, key=lambda k: (-fused[k], k)):
            chunk, terms = lookup[key], snapshot.terms[key]
            if any(
                p.document_id == chunk.document_id
                and p.page == chunk.page
                and len(terms & snapshot.terms[p.id]) / max(1, len(terms | snapshot.terms[p.id]))
                > 0.92
                for p in selected
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
    provider = "none"

    async def rank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks


class LocalReranker:
    provider = "local"

    def __init__(
        self, model: str, *, device: str = "cpu", batch_size: int = 4, max_length: int = 512
    ) -> None:
        self.model_name = model
        self.device = device
        self.batch_size, self.max_length = batch_size, max_length
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def rank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._rank, query, chunks)

    def _rank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        with INFERENCE_LOCK:
            return self._rank_locked(query, chunks)

    def _rank_locked(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                options: dict[str, Any] = {
                    "trust_remote_code": False,
                    "max_length": self.max_length,
                }
                if self.device != "auto":
                    options["device"] = self.device
                self._model = CrossEncoder(self.model_name, **options)
            scores = self._model.predict(
                [(query, c.text) for c in chunks],
                batch_size=self.batch_size,
                show_progress_bar=False,
            ).tolist()
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
