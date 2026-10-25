import json

import httpx
import pytest
from pydantic import SecretStr

from research_assist.config import Settings
from research_assist.domain.errors import ProviderError
from research_assist.domain.models import RetrievalPlan
from research_assist.infrastructure.embeddings import APIEmbeddings
from research_assist.infrastructure.llm import HTTPLanguageModel, usage_scope


def valid_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({"queries": ["RAG methods"], "facets": []})},
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        },
    )


async def generate(llm: HTTPLanguageModel) -> RetrievalPlan:
    return await llm.generate(
        task="retrieval_plan",
        instruction="Return queries",
        payload={"question": "RAG methods"},
        schema=RetrievalPlan,
        fallback=lambda: RetrievalPlan(queries=["OFFLINE FALLBACK"]),
    )


async def test_openai_schema_and_usage(settings: Settings) -> None:
    settings.llm_provider = "openai"
    settings.openai_api_key = SecretStr("unit-secret")
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return valid_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        llm = HTTPLanguageModel(settings, client)
        with usage_scope() as usage:
            result = await generate(llm)
        assert result.queries == ["RAG methods"]
        assert (usage.calls, usage.input_tokens, usage.output_tokens) == (1, 12, 8)
    body = json.loads(seen[0].content)
    assert body["response_format"]["json_schema"]["strict"]
    assert "facets" in body["response_format"]["json_schema"]["schema"]["required"]
    assert seen[0].headers["authorization"] == "Bearer unit-secret"


async def test_schema_repair_once(settings: Settings) -> None:
    settings.llm_provider = "openai"
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": '{"queries":[],"extra":true}'}}]}
            )
        return valid_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with usage_scope() as usage:
            result = await generate(HTTPLanguageModel(settings, client))
    assert result.queries and usage.calls == 2
    assert [e["status"] for e in usage.provider_events if e["kind"] == "http_attempt"] == [200, 200]
    repair = next(e for e in usage.provider_events if e["kind"] == "schema_validation_failed")
    assert "extra_forbidden" in repair["error_types"]
    assert "unit-secret" not in json.dumps(usage.provider_events)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
    ],
)
async def test_malformed_success_never_falls_back(
    settings: Settings, payload: dict[str, object]
) -> None:
    settings.llm_provider = "openai"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(ProviderError):
            await generate(HTTPLanguageModel(settings, client))


async def test_repair_exhaustion_never_falls_back(settings: Settings) -> None:
    settings.llm_provider = "openai"
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"choices": [{"message": {"content": "invalid JSON"}}]})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with usage_scope() as usage, pytest.raises(ProviderError, match="hợp đồng"):
            await generate(HTTPLanguageModel(settings, client))
    assert usage.calls == 2


async def test_http_error_does_not_expose_key_or_response(settings: Settings) -> None:
    settings.llm_provider = "openai"
    transport = httpx.MockTransport(
        lambda _: httpx.Response(401, text="PRIVATE RAW RESPONSE unit-secret")
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ProviderError) as error:
            await generate(HTTPLanguageModel(settings, client))
    assert "401" in str(error.value) and "unit-secret" not in str(error.value)
    assert "PRIVATE" not in str(error.value)


async def test_retry_is_bounded_and_budgeted(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.llm_provider = "openai"

    async def immediate(_: float) -> None:
        pass

    monkeypatch.setattr("research_assist.infrastructure.llm.asyncio.sleep", immediate)
    transport = httpx.MockTransport(lambda _: httpx.Response(429, text="quota"))
    async with httpx.AsyncClient(transport=transport) as client:
        with usage_scope() as usage, pytest.raises(ProviderError):
            await generate(HTTPLanguageModel(settings, client))
    assert usage.calls == 3


async def test_configurable_retries_record_wait_and_remain_budgeted(settings, monkeypatch):
    settings.llm_provider = "openai"
    settings.llm_retry_attempts = 5
    settings.llm_retry_initial_seconds = 5
    waits = []

    async def immediate(delay):
        waits.append(delay)

    monkeypatch.setattr("research_assist.infrastructure.llm.asyncio.sleep", immediate)
    transport = httpx.MockTransport(lambda _: httpx.Response(503))
    async with httpx.AsyncClient(transport=transport) as client:
        with usage_scope() as usage, pytest.raises(ProviderError, match="HTTP 503"):
            await generate(HTTPLanguageModel(settings, client))
        assert usage.calls == 5 and len(waits) == 4
        for index, delay in enumerate(waits):
            assert 5 * 2**index <= delay <= min(60, 6.25 * 2**index)
        retry_events = [e for e in usage.provider_events if e["kind"] == "retry_scheduled"]
        assert len(retry_events) == 4
        settings.llm_max_calls = 3
        with usage_scope() as limited, pytest.raises(ProviderError, match="ngân sách số lần"):
            await generate(HTTPLanguageModel(settings, client))
        assert limited.calls == 3


async def test_gemini_payload_thought_filter_and_usage(settings: Settings) -> None:
    settings.llm_provider = "gemini"
    settings.gemini_api_key = SecretStr("unit-gemini")
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {"thought": True, "text": "hidden reasoning"},
                                {"text": '{"queries":["RAG methods"],"facets":[]}'},
                            ]
                        },
                    }
                ],
                "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 7},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with usage_scope() as usage:
            result = await generate(HTTPLanguageModel(settings, client))
    assert result.queries == ["RAG methods"]
    assert usage.input_tokens == 9 and usage.output_tokens == 7
    assert seen[0].headers["x-goog-api-key"] == "unit-gemini"
    assert "responseJsonSchema" in json.loads(seen[0].content)["generationConfig"]


async def test_embedding_api_orders_by_index(settings: Settings) -> None:
    settings.embedding_model = "test-embedding"
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0, 2]},
                    {"index": 0, "embedding": [2, 0]},
                ]
            },
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        result = await APIEmbeddings(settings, client).embed(["a", "b"])
    assert result == [[1, 0], [0, 1]]


async def test_embedding_api_rejects_duplicate_indices(settings: Settings) -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [1, 0]},
                    {"index": 0, "embedding": [1, 0]},
                ]
            },
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ProviderError):
            await APIEmbeddings(settings, client).embed(["a", "b"])


def gemini_response(reason: str = "STOP") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "finishReason": reason,
                    "content": {"parts": [{"text": '{"queries":["RAG methods"],"facets":[]}'}]},
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 9,
                "candidatesTokenCount": 7,
                "thoughtsTokenCount": 13,
            },
        },
    )


@pytest.mark.parametrize(
    "model,budget,sent",
    [
        ("gemini-2.5-flash", 1024, True),
        ("models/gemini-2.5-flash", 0, True),
        ("gemini-2.5-flash-lite", -1, True),
        ("gemini-2.5-pro", 0, False),
    ],
)
async def test_gemini_thinking_budget_is_model_specific(
    settings: Settings, model: str, budget: int, sent: bool
) -> None:
    settings.llm_provider = "gemini"
    settings.llm_model = model
    settings.gemini_thinking_budget = budget
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["generationConfig"])
        return gemini_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with usage_scope() as usage:
            await generate(HTTPLanguageModel(settings, client))
    assert ("thinkingConfig" in seen[0]) is sent
    if sent:
        assert seen[0]["thinkingConfig"]["thinkingBudget"] == budget
    assert usage.output_tokens == 20


@pytest.mark.parametrize("recover", [True, False])
async def test_gemini_token_exhaustion_regenerates_only_once(
    settings: Settings, recover: bool
) -> None:
    settings.llm_provider = "gemini"
    limits = []

    def handle(request: httpx.Request) -> httpx.Response:
        limits.append(json.loads(request.content)["generationConfig"]["maxOutputTokens"])
        return gemini_response("STOP" if recover and len(limits) == 2 else "MAX_TOKENS")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with usage_scope() as usage:
            if recover:
                assert (await generate(HTTPLanguageModel(settings, client))).queries
            else:
                with pytest.raises(ProviderError, match="MAX_TOKENS"):
                    await generate(HTTPLanguageModel(settings, client))
    assert limits == [5000, 10000]
    assert usage.calls == 2 and usage.output_tokens == 40


async def test_gemini_token_cap_and_call_budget(settings: Settings) -> None:
    settings.llm_provider = "gemini"
    settings.llm_max_output_tokens = 12000
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: gemini_response("MAX_TOKENS"))
    ) as client:
        with usage_scope() as usage, pytest.raises(ProviderError, match="MAX_TOKENS"):
            await generate(HTTPLanguageModel(settings, client))
        assert usage.calls == 1
        settings.llm_max_output_tokens = 5000
        with usage_scope() as usage, pytest.raises(ProviderError, match="ngân sách số lần"):
            usage.calls = settings.llm_max_calls - 1
            await generate(HTTPLanguageModel(settings, client))
        assert usage.calls == settings.llm_max_calls


@pytest.mark.parametrize(
    "sequence,expected_limits,success",
    [
        (["MAX_TOKENS", "invalid", "STOP"], [5000, 10000, 10000], True),
        (["invalid", "MAX_TOKENS", "STOP"], [5000, 5000, 10000], True),
        (["MAX_TOKENS", "invalid", "MAX_TOKENS"], [5000, 10000, 10000], False),
    ],
)
async def test_json_repair_retains_grown_output_limit(settings, sequence, expected_limits, success):
    settings.llm_provider = "gemini"
    limits = []

    def handle(request):
        limits.append(json.loads(request.content)["generationConfig"]["maxOutputTokens"])
        reason = sequence[len(limits) - 1]
        if reason != "invalid":
            return gemini_response(reason)
        body = gemini_response().json()
        body["candidates"][0]["content"]["parts"][0]["text"] = "invalid JSON"
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with usage_scope() as usage:
            if success:
                assert (await generate(HTTPLanguageModel(settings, client))).queries
            else:
                with pytest.raises(ProviderError, match="MAX_TOKENS"):
                    await generate(HTTPLanguageModel(settings, client))
        assert usage.calls == 3
    assert limits == expected_limits


@pytest.mark.parametrize("reason", ["SAFETY", "RECITATION", "PRIVATE RAW unit-secret"])
async def test_gemini_stop_reason_is_safe_and_not_retried(settings: Settings, reason: str) -> None:
    settings.llm_provider = "gemini"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: gemini_response(reason))
    ) as client:
        with usage_scope() as usage, pytest.raises(ProviderError) as error:
            await generate(HTTPLanguageModel(settings, client))
    assert usage.calls == 1
    assert (reason if reason in {"SAFETY", "RECITATION"} else "UNKNOWN") in str(error.value)
    assert "PRIVATE" not in str(error.value)


@pytest.mark.parametrize("reason", ["SAFETY", "PRIVATE RAW unit-secret"])
async def test_gemini_blocked_prompt_without_candidates(settings: Settings, reason: str) -> None:
    settings.llm_provider = "gemini"
    response = httpx.Response(200, json={"promptFeedback": {"blockReason": reason}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        with usage_scope() as usage, pytest.raises(ProviderError, match="chặn đầu vào") as error:
            await generate(HTTPLanguageModel(settings, client))
    assert usage.calls == 1 and "PRIVATE" not in str(error.value)
