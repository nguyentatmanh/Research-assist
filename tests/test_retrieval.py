import math

import pytest

from research_assist.application.documents import DocumentService
from research_assist.config import Settings
from research_assist.domain.errors import ProviderError
from research_assist.infrastructure.embeddings import NoEmbeddings, validate_vectors
from research_assist.infrastructure.repository import Repository
from research_assist.infrastructure.search import HybridSearch, bm25_scores


def test_bm25_prioritizes_relevant_source_and_zero_is_excluded() -> None:
    scores = bm25_scores(
        "retrieval evidence", ["retrieval retrieval evidence", "tomatoes grow in warm weather"]
    )
    assert scores[0] > 0 and scores[1] == 0
    assert bm25_scores("", ["anything"]) == [0]


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
