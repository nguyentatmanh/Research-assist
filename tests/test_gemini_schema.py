import json

import httpx
import pytest
from pydantic import BaseModel, Field

from research_assist.config import Settings
from research_assist.domain.errors import ProviderError
from research_assist.domain.models import (
    AgentDecision,
    FactCheckOutput,
    ReaderOutput,
    SearchCompletion,
    SynthesisOutput,
    WriterOutput,
)
from research_assist.infrastructure.llm import (
    HTTPLanguageModel,
    gemini_schema,
    strict_schema,
    usage_scope,
)


@pytest.mark.parametrize(
    "output", [SearchCompletion, ReaderOutput, SynthesisOutput, FactCheckOutput, WriterOutput]
)
def test_all_worker_wire_schemas_keep_types_and_remove_nested_bounds(output) -> None:
    schema = strict_schema(AgentDecision[output])
    original = json.dumps(schema)
    wire = gemini_schema(schema)
    assert json.dumps(schema) == original
    assert wire["required"] == schema["required"]
    assert wire["properties"]["action"] == schema["properties"]["action"]
    assert wire["additionalProperties"] is False
    assert wire["$defs"].keys() == schema["$defs"].keys()
    for keyword in ("maxItems", "minItems", "minimum", "maximum", "maxLength", "minLength"):
        assert f'"{keyword}":' not in json.dumps(wire)


def test_schema_property_names_are_preserved() -> None:
    schema = {"type": "object", "properties": {"minimum": {"type": "number", "minimum": 1}}}
    assert gemini_schema(schema)["properties"]["minimum"] == {"type": "number"}


class BoundedOutput(BaseModel):
    items: list[int] = Field(max_length=1)


async def test_full_local_bounds_still_reject_oversized_model_output(settings: Settings) -> None:
    settings.llm_provider = "gemini"
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        wire = body["generationConfig"]["responseJsonSchema"]
        assert "maxItems" not in json.dumps(wire)
        # The original schema in instructions and local validation retain the bound.
        assert "maxItems" in body["systemInstruction"]["parts"][0]["text"]
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"finishReason": "STOP", "content": {"parts": [{"text": '{"items":[1,2]}'}]}}
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with usage_scope() as usage, pytest.raises(ProviderError, match="hợp đồng"):
            await HTTPLanguageModel(settings, client).generate(
                task="reader_step",
                instruction="Return at most one item",
                payload={},
                schema=BoundedOutput,
                fallback=lambda: BoundedOutput(items=[]),
            )
        assert usage.calls == len(seen) == 2
        assert all(
            "too_long" in event["error_types"]
            for event in usage.provider_events
            if event["kind"] == "schema_validation_failed"
        )
