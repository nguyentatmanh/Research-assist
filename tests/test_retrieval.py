import math

import pytest

from research_assist.agents.retrieval import RetrievalAgent
from research_assist.application.documents import DocumentService
from research_assist.config import Settings
from research_assist.domain.errors import ProviderError
from research_assist.domain.models import (
    IndexedChunk,
    QuestionAspect,
    QuestionPlan,
    ResearchRequest,
    RetrievedChunk,
)
from research_assist.domain.text import bm25_tokens, extract_literal_anchors, tokens
from research_assist.infrastructure.embeddings import NoEmbeddings, validate_vectors
from research_assist.infrastructure.llm import DemoLanguageModel
from research_assist.infrastructure.repository import Repository
from research_assist.infrastructure.search import HybridSearch, bm25_scores


@pytest.mark.parametrize("candidate_limit", [1, 2])
async def test_retry_reads_deeper_hits_even_when_reranker_prefers_old_hits(chunk, candidate_limit):
    old = chunk.model_copy(update={"id": "ch_old"})
    fresh = chunk.model_copy(update={"id": "ch_new"})

    class Search:
        limits = []

        async def search(self, queries, document_ids, top_k, mode):
            self.limits.append(top_k)
            return [old, fresh][:top_k], []

    class Reranker:
        async def rank(self, query, chunks):
            return sorted(chunks, key=lambda c: c.id != old.id)

    search = Search()
    agent = RetrievalAgent(DemoLanguageModel(), search, Reranker(), candidate_limit=candidate_limit)
    request = ResearchRequest(question="Explain retrieval evidence", top_k=2)
    plan = await agent.prepare(request)
    _, first, _ = await agent.run(request, [], plan)
    trace, second, _ = await agent.run(request, [], plan, {"rq_1"}, {old.id})
    assert first[0].id == old.id
    assert second[0].id == fresh.id
    assert search.limits == [candidate_limit, candidate_limit + 1]
    assert trace.trace[0]["previously_seen_count"] == 1


def test_bm25_prioritizes_relevant_source_and_zero_is_excluded() -> None:
    scores = bm25_scores(
        "retrieval evidence", ["retrieval retrieval evidence", "tomatoes grow in warm weather"]
    )
    assert scores[0] > 0 and scores[1] == 0
    assert bm25_scores("", ["anything"]) == [0]


def test_bm25_lexes_numbers_without_changing_reader_tokenization() -> None:
    assert bm25_tokens("6 8 28.4 42.5% 5 % d_model d_ff gpt-oss-120b v1.2") == [
        "6",
        "8",
        "28.4",
        "42.5%",
        "42.5",
        "5%",
        "5",
        "d_model",
        "d_ff",
        "gpt-oss-120b",
        "v1.2",
    ]
    assert tokens("6 8 28.4") == ["28"]
    assert extract_literal_anchors("6 8 28.4 42.5% 5% v1.2") == {"28.4", "42.5%", "5%", "v1.2"}


def test_bilingual_bm25_only_expands_known_query_phrases() -> None:
    corpus = ["The self-attention layer uses positional encoding.", "Attention is all you need."]
    relevant, generic = bm25_scores("tự chú ý", corpus)
    assert relevant > 0 and generic == 0
    assert bm25_scores("mã hóa vị trí", corpus)[0] > 0
    assert bm25_scores("hồi quy và kiến trúc", corpus) == [0, 0]
    assert bm25_scores("mã hóa vị trí positional encoding", corpus) == bm25_scores(
        "positional encoding", corpus
    )


class DistractorEmbedder:
    fingerprint: str | None = "test:literal:v1"

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        return [[1, 0] if query or "distractor" in text else [0, 1] for text in texts]


class DistractorReranker:
    async def rank(self, query, chunks):
        return sorted(chunks, key=lambda chunk: "distractor" not in chunk.text)


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
async def test_search_and_agent_preserve_exact_literals_with_lower_scoring_sources(settings, mode):
    repository = Repository(settings.data_dir)
    embedder = DistractorEmbedder()
    service = DocumentService(repository, embedder, settings)
    numeric, _ = await service.ingest(
        b"The decimal result 28.4 and the v1.2 release used d_model.", "measurements.txt"
    )
    table, _ = await service.ingest(
        b"Dataset | Accuracy (%) | Count\nStudy A | 42.5 | 7\nStudy B | 5 | 10", "table.txt"
    )
    for n in range(3):
        await service.ingest(
            (
                f"distractor {n}: decimal result 28.40 and v1.20 refer to different values. "
                "A dataset Count row contains 5 and 42.5 observations."
            ).encode(),
            f"distractor-{n}.txt",
        )
    search = HybridSearch(repository, embedder, settings)
    agent = RetrievalAgent(DemoLanguageModel(), search, DistractorReranker(), candidate_limit=3)
    for literal, expected in (
        ("28.4", numeric),
        ("v1.2", numeric),
        ("d_model", numeric),
        ("42.5%", table),
        ("5%", table),
    ):
        query = f"Find the exact decimal result for {literal} in the dataset"
        raw, _ = await search.search([query], [], 1, mode)
        assert raw[0].document_id == expected.id
        request = ResearchRequest(question=query, retrieval_mode=mode, top_k=2)
        plan = QuestionPlan(aspects=[QuestionAspect(text=query, query=query)])
        _, selected, _ = await agent.run(request, [], plan)
        assert selected[0].document_id == expected.id

    # Terms in different sources remain usable. The original unanchored lane
    # must not fill both slots before the explicit candidates are considered.
    request = ResearchRequest(question="Find the distractor dataset", retrieval_mode=mode, top_k=2)
    query = "Find the model identifier d_model and the table value 42.5%"
    plan = QuestionPlan(aspects=[QuestionAspect(text=query, query=query)])
    _, selected, _ = await agent.run(request, [], plan)
    assert {chunk.document_id for chunk in selected} == {numeric.id, table.id}

    # A second real requirement keeps a slot even when the first has split literals.
    plan.aspects.append(QuestionAspect(text=request.question, query=request.question))
    _, selected, _ = await agent.run(request, [], plan)
    assert len(selected) == 2
    assert any(chunk.document_id in {numeric.id, table.id} for chunk in selected)
    assert any("distractor" in chunk.filename for chunk in selected)


async def test_same_page_duplicate_windows_keep_distinct_requested_numbers(
    settings, document, chunk
):
    repository = Repository(settings.data_dir)
    chunks = []
    fields = chunk.model_dump(exclude={"score", "lexical_score", "semantic_score", "rerank_score"})
    for i, score in enumerate(("28.4", "28.6"), 1):
        text = f"Machine translation results for the experiment. Reported score {score} BLEU."
        chunks.append(
            IndexedChunk(
                **{**fields, "id": f"ch_{i}", "text": text, "end": chunk.start + len(text)}
            )
        )
    repository.add_document(document.model_copy(update={"chunk_count": 2}), chunks, "study.pdf")
    search = HybridSearch(repository, NoEmbeddings(), settings)
    selected, _ = await search.search(["Compare 28.4 and 28.6 BLEU"], [document.id], 2, "lexical")
    assert {item.id for item in selected} == {"ch_1", "ch_2"}
    # A value from a different document cannot replace the matching window here.
    other = document.model_copy(update={"id": "doc_other", "sha256": "other"})
    other_chunk = chunks[0].model_copy(update={"id": "ch_0", "document_id": other.id})
    repository.add_document(other, [other_chunk], "other.pdf")
    selected, _ = await search.search(
        ["Compare 28.4 and 28.6 BLEU"], [document.id, other.id], 3, "lexical"
    )
    assert {item.id for item in selected} == {"ch_0", "ch_1", "ch_2"}


async def test_expand_context_restores_start_of_result_without_full_neighbor(
    settings, document, chunk
):
    repository = Repository(settings.data_dir)
    introduction = " On WMT 2014 English-to-German the big Transformer"
    continuation = " achieved 28.4 BLEU."
    first = "Prior discussion. " * 100 + introduction
    text = first + continuation + " Further discussion." * 55
    split = len(first) - len("Transformer")
    fields = chunk.model_dump(exclude={"score", "lexical_score", "semantic_score", "rerank_score"})
    previous = IndexedChunk(
        **{
            **fields,
            "id": "ch_prev",
            "start": 0,
            "end": len(first),
            "text": first,
        }
    )
    seed = RetrievedChunk(
        **{
            **fields,
            "id": "ch_next",
            "ordinal": 1,
            "start": split,
            "end": len(text),
            "text": text[split:],
            "score": 1.0,
        }
    )
    repository.add_document(
        document.model_copy(update={"chunk_count": 2}),
        [previous, IndexedChunk(**seed.model_dump(include=set(IndexedChunk.model_fields)))],
        "study.pdf",
    )
    frozen = await HybridSearch(repository, NoEmbeddings(), settings).freeze([document.id])
    (expanded,) = await frozen.expand_context([seed], [document.id])
    assert len(text) > 2600 >= len(expanded.text)
    assert expanded.id == seed.id and expanded.start < seed.start
    assert introduction + continuation in expanded.text
    assert expanded.text == text[expanded.start : expanded.end]


async def test_one_bilingual_passage_survives_many_source_language_hits(settings):
    repository = Repository(settings.data_dir)
    service = DocumentService(repository, NoEmbeddings(), settings)
    english, _ = await service.ingest(
        b"The decoder uses masking to prevent attending to subsequent positions.", "paper-en.txt"
    )
    for n in range(4):
        await service.ingest(
            f"distractor {n}: Cơ chế che vị trí tương lai cho hệ thống vật lý.".encode(),
            f"paper-vi-{n}.txt",
        )
    search = HybridSearch(repository, NoEmbeddings(), settings)
    question = "cơ chế che vị trí tương lai"
    raw, _ = await search.search([question], [], 2, "lexical")
    assert raw[0].document_id != english.id and raw[1].document_id == english.id
    agent = RetrievalAgent(DemoLanguageModel(), search, DistractorReranker())
    request = ResearchRequest(question=question, retrieval_mode="lexical", top_k=2)
    _, selected, _ = await agent.run(request, [])
    assert selected[0].document_id != english.id and selected[1].document_id == english.id


@pytest.mark.parametrize(
    "vectors,count",
    [
        ([[0, 0]], 1),
        ([[math.nan, 1]], 1),
        ([[math.inf, 1]], 1),
        ([[True, 1]], 1),
        ([["x", 1]], 1),
        ([[1, 2], [1]], 2),
        ([[1, 2]], 2),
        ([], 1),
    ],
)
def test_embedding_boundary_rejects_bad_vectors(vectors: object, count: int) -> None:
    with pytest.raises(ProviderError):
        validate_vectors(vectors, count)


def test_vectors_normalized() -> None:
    assert validate_vectors([[3, 4]], 1) == [[0.6, 0.8]]


def test_large_finite_vector_normalizes_without_overflow() -> None:
    vectors = validate_vectors([[1e308, 1e308]], 1)
    assert all(math.isfinite(value) for value in vectors[0])
    assert math.isclose(sum(value**2 for value in vectors[0]), 1.0)


class TinyEmbedder:
    fingerprint: str | None = "test:science:v1"

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        return [[1.0, 0.0] if query or "galaxies" in text else [0.0, 1.0] for text in texts]


async def test_dense_retrieval_adds_source_without_keyword_overlap(settings: Settings) -> None:
    repository = Repository(settings.data_dir)
    embedder = TinyEmbedder()
    service = DocumentService(repository, embedder, settings)
    science, _ = await service.ingest(
        b"Distant galaxies contain billions of stars and planetary systems.", "space.txt"
    )
    await service.ingest(
        b"Tomatoes grow in warm weather with enough water and suitable soil.", "garden.txt"
    )
    search = HybridSearch(repository, embedder, settings)
    lexical, _ = await search.search(["astronomy"], [], 4, "lexical")
    hybrid, warnings = await search.search(["astronomy"], [], 4, "hybrid")
    assert not lexical
    assert hybrid[0].document_id == science.id and hybrid[0].semantic_score == 1
    assert not warnings


async def test_scope_cannot_leak_other_documents(settings: Settings) -> None:
    repository = Repository(settings.data_dir)
    service = DocumentService(repository, NoEmbeddings(), settings)
    selected, _ = await service.ingest(
        b"RAG uses external evidence to support generated answers.", "selected.txt"
    )
    await service.ingest(
        b"RAG uses private evidence that should be outside this scope.", "other.txt"
    )
    results, warnings = await HybridSearch(repository, NoEmbeddings(), settings).search(
        ["RAG evidence"], [selected.id], 8, "hybrid"
    )
    assert all(result.document_id == selected.id for result in results)
    assert warnings and results


async def test_changed_embedding_fingerprint_does_not_mix_vector_spaces(settings: Settings) -> None:
    repository = Repository(settings.data_dir)
    embedder = TinyEmbedder()
    service = DocumentService(repository, embedder, settings)
    await service.ingest(
        b"Distant galaxies contain billions of stars and planetary systems.", "space.txt"
    )
    embedder.fingerprint = "test:new-model:alternate"
    results, warnings = await HybridSearch(repository, embedder, settings).search(
        ["astronomy"], [], 4, "hybrid"
    )
    assert not results
    assert "Reindex" in warnings[0]


async def test_embedding_dimension_change_requires_reindex(settings: Settings) -> None:
    class WrongDimension(TinyEmbedder):
        async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
            return [[1.0, 0.0, 0.0] for _ in texts] if query else await super().embed(texts)

    repository = Repository(settings.data_dir)
    embedder = WrongDimension()
    await DocumentService(repository, embedder, settings).ingest(
        b"Distant galaxies contain billions of stars and planetary systems.", "space.txt"
    )
    with pytest.raises(ProviderError, match="Chiều"):
        await HybridSearch(repository, embedder, settings).search(["astronomy"], [], 4, "hybrid")


@pytest.mark.parametrize(
    "revision,expected", [("same-checkpoint", True), ("other-checkpoint", False)]
)
async def test_vector_format_labels_preserve_checkpoint_identity(
    settings: Settings, revision: str, expected: bool
) -> None:
    repository = Repository(settings.data_dir)
    embedder = TinyEmbedder()
    embedder.fingerprint = "sentence-transformers:science:e5:rev:same-checkpoint:512:v7"
    document, _ = await DocumentService(repository, embedder, settings).ingest(
        b"Distant galaxies contain billions of stars and planetary systems.", "space.txt"
    )
    stored = repository.chunks([document.id])[0].vector_model
    embedder.fingerprint = f"sentence-transformers:science:e5:rev:{revision}:512:normalized"
    results, warnings = await HybridSearch(repository, embedder, settings).search(
        ["astronomy"], [document.id], 4, "hybrid"
    )
    assert bool(results) is expected
    assert bool(warnings) is not expected
    assert repository.chunks([document.id])[0].vector_model == stored
