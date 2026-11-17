import json

import httpx
import pytest

from research_assist.domain.errors import CapacityError, DocumentError, ProviderError
from research_assist.infrastructure.openalex import OpenAlexLibrary, normalize_doi


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  DOI: 10.1000/Article.1  ", "10.1000/article.1"),
        ("HTTPS://DX.DOI.ORG/10.1000/Article%2F1?source=copy", "10.1000/article/1"),
        ("10.1000/some?symbol", "10.1000/some?symbol"),
        ("https://doi.org.example/10.1000/article", None),
        ("10.1000/x/../article", None),
        ("10.1000/\ud800", None),
        ("topic without an identifier", None),
    ],
)
def test_normalize_doi(value: str, expected: str | None) -> None:
    assert normalize_doi(value) == expected


async def test_keyword_search_returns_metadata_and_reconstructed_abstract() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.scheme == "https"
        assert request.url.host == "api.openalex.org"
        assert request.url.path == "/works"
        assert request.url.params["search"] == "tiếng Việt retrieval"
        assert request.url.params["per_page"] == "5"
        assert "abstract_inverted_index" in request.url.params["select"]
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "title": " Bilingual  Retrieval ",
                        "doi": "https://doi.org/10.1000/Study",
                        "publication_date": "2025-07-12",
                        "authorships": [
                            {"author": {"display_name": " Linda   Nguyen "}},
                            {"author": None},
                        ],
                        "abstract_inverted_index": {
                            "works.": [3],
                            "and": [1],
                            "Retrieval": [0, 2],
                        },
                        "open_access": {"oa_url": "https://other.example/paper.pdf"},
                    },
                    {
                        "id": "https://openalex.org/W234",
                        "title": "Paper without DOI",
                        "publication_year": 2024,
                    },
                    {"id": "https://other.example/W456", "title": "Unknown record ID"},
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        first, second = await OpenAlexLibrary(client).search(" tiếng Việt   retrieval ")
    assert first.paper_id == "W123" and first.source == "openalex"
    assert first.title == "Bilingual Retrieval" and first.authors == ["Linda Nguyen"]
    assert first.abstract == "Retrieval and Retrieval works."
    assert first.doi == "10.1000/study" and first.url == "https://doi.org/10.1000/study"
    assert first.published == "2025-07-12" and not first.import_available
    assert second.abstract is None and second.doi is None
    assert second.url == "https://openalex.org/W234" and second.published == "2024"
    assert not second.import_available


async def test_bare_doi_uses_exact_lookup_and_optional_bearer_authentication() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/works/doi:10.1000/articles/v2"
        assert "search" not in request.url.params
        assert "api_key" not in request.url.params
        assert request.headers["authorization"] == "Bearer fixture-key"
        return httpx.Response(
            200,
            json={"id": "https://openalex.org/W123", "title": "A work"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await OpenAlexLibrary(client, "fixture-key").search("10.1000/articles/v2")
    assert [p.paper_id for p in result] == ["W123"]


@pytest.mark.parametrize(
    ("query", "status", "body"),
    [("missing topic", 200, {"results": []}), ("10.1000/missing", 404, {"error": "Not found"})],
)
async def test_missing_search_results_and_doi_are_empty(
    query: str, status: int, body: dict[str, object]
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json=body))
    ) as client:
        assert await OpenAlexLibrary(client).search(query) == []


@pytest.mark.parametrize(
    "abstract",
    [
        {"Bad": [40000000]},
        {"Bad": [0], "collision": [0]},
        {"Broken": [0, 2]},
        {"Wrong type": [True]},
        {"word": "one"},
        {"lo" * 8100: [0]},
    ],
    ids=["too-large", "collision", "gap", "boolean", "wrong-type", "long-text"],
)
async def test_malformed_abstract_is_omitted_without_discarding_work(abstract: object) -> None:
    payload = {
        "id": "https://openalex.org/W123",
        "title": "Paper",
        "abstract_inverted_index": abstract,
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"results": [payload]}))
    ) as client:
        (paper,) = await OpenAlexLibrary(client).search("missing metadata")
    assert paper.abstract is None


async def test_unpaired_surrogates_in_provider_metadata_do_not_break_scout_json() -> None:
    payload = {
        "results": [
            {"id": "https://openalex.org/W1", "title": "\ud800"},
            {
                "id": "https://openalex.org/W2",
                "title": "Valid text",
                "doi": "10.1000/\ud800",
                "authorships": [{"author": {"display_name": "\ud800"}}],
                "abstract_inverted_index": {"\ud800": [0]},
            },
        ]
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=json.dumps(payload).encode("ascii"))
        )
    ) as client:
        (paper,) = await OpenAlexLibrary(client).search("research topic")
    assert paper.paper_id == "W2" and not paper.authors
    assert paper.doi is None and paper.abstract is None
    assert paper.model_dump_json().encode("utf-8")


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(429), CapacityError),
        (httpx.Response(200, content=b"{invalid"), ProviderError),
        (httpx.Response(200, json={"results": {"id": "W123"}}), ProviderError),
        (httpx.Response(302, headers={"location": "https://redirect.example"}), ProviderError),
        (httpx.Response(200, content=b"x" * (524288 + 1)), DocumentError),
    ],
    ids=["rate-limit", "invalid-json", "invalid-envelope", "redirect", "response-size"],
)
async def test_bad_responses_fail_without_an_external_redirect(
    response: httpx.Response, error: type[Exception]
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.openalex.org"
        return response

    async with httpx.AsyncClient(
        follow_redirects=True, transport=httpx.MockTransport(respond)
    ) as client:
        with pytest.raises(error):
            await OpenAlexLibrary(client, "fixture-key").search("research topic")
