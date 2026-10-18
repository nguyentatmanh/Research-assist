import json

import httpx
import pytest
from pydantic import SecretStr

from research_assist.bootstrap import build_container
from research_assist.config import Settings
from research_assist.domain.models import ResearchRequest


@pytest.mark.parametrize("profile,expected_calls", [("academic", 4), ("balanced", 3)])
@pytest.mark.parametrize("complex_question", [False, True])
async def test_five_agents_call_real_http_adapter_without_demo_fallback(
    settings: Settings, profile: str, expected_calls: int, complex_question: bool
) -> None:
    settings.llm_provider = "openai"
    settings.workflow_profile = profile
    settings.openai_api_key = SecretStr("test-only-key")
    tasks = []
    question = (
        "Compare RAG retrieval and source evidence"
        if complex_question
        else "Explain how RAG uses evidence"
    )
    expected_calls += int(complex_question)

    def provider(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        task = body["response_format"]["json_schema"]["name"]
        payload = json.loads(body["messages"][1]["content"])
        tasks.append(task)
        if task == "question_plan":
            output = {
                "aspects": [
                    {"text": "Retrieval", "query": "RAG retrieval"},
                    {"text": "Evidence", "query": "RAG evidence"},
                ]
            }
        elif task == "read_evidence":
            chunk = payload["source_units"][0]
            output = {
                "evidence": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "unit_id": chunk["unit_id"],
                        "quote": "",
                        "relevance": "Relevant to RAG",
                    }
                ],
                "gaps": [],
            }
        elif task == "synthesize_claims":
            evidence = payload["evidence"][0]
            output = {
                "claims": [{"text": evidence["quote"], "evidence_ids": [evidence["id"]]}],
                "gaps": [],
            }
        elif task == "check_claims":
            claim = payload["claims"][0]
            output = {
                "checks": [
                    {
                        "claim_id": claim["id"],
                        "verdict": "supported",
                        "explanation": "Direct quotation",
                        "evidence_ids": claim["evidence_ids"],
                    }
                ]
            }
        elif task == "write_report":
            claim = payload["verified_claims"][0]
            output = {
                "title": "RAG overview",
                "sections": [{"heading": "Evidence", "blocks": [{"claim_id": claim["id"]}]}],
            }
        else:
            raise AssertionError(f"Unexpected provider task: {task}")
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    container = build_container(settings, httpx.MockTransport(provider))
    try:
        await container.documents.ingest(
            b"RAG combines retrieval with generation and source evidence.", "paper.txt"
        )
        record = container.runs.submit(ResearchRequest(question=question, max_rounds=1))
        await container.runs.tasks[record.id]
        finished = container.repository.run(record.id)
        assert finished.status == "completed", finished.error
        assert finished.result is not None
        assert finished.result.provider == "openai"
        assert finished.result.llm_calls == expected_calls
        assert (finished.result.input_tokens, finished.result.output_tokens) == (
            20 * expected_calls,
            10 * expected_calls,
        )
        expected_tasks = [
            "read_evidence",
            "synthesize_claims",
            "check_claims",
            "write_report",
        ]
        expected_tasks = expected_tasks if profile == "academic" else expected_tasks[:3]
        if complex_question:
            expected_tasks.insert(0, "question_plan")
        assert tasks == expected_tasks
        assert "[1]" in finished.result.markdown
        repeated = container.runs.submit(ResearchRequest(question=question, max_rounds=1))
        await container.runs.tasks[repeated.id]
        reused = container.repository.run(repeated.id)
        assert reused.result is not None, reused.error
        assert reused.result.llm_calls == 0
        assert reused.result.cache_hits == expected_calls
        assert reused.result.markdown == finished.result.markdown
    finally:
        await container.close()


async def test_budget_failure_trace_preserves_consumed_tokens(settings):
    settings.llm_provider = "openai"
    settings.openai_api_key = SecretStr("test-key")
    settings.llm_max_total_tokens = 1000

    def provider(request):
        body = json.loads(request.content)
        assert body["response_format"]["json_schema"]["name"] == "read_evidence"
        unit = json.loads(body["messages"][1]["content"])["source_units"][0]
        output = {
            "evidence": [
                {
                    "chunk_id": unit["chunk_id"],
                    "unit_id": unit["unit_id"],
                    "quote": "",
                    "relevance": "Relevant",
                }
            ],
            "gaps": [],
        }
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}],
                "usage": {"prompt_tokens": 990, "completion_tokens": 10},
            },
        )

    container = build_container(settings, httpx.MockTransport(provider))
    try:
        await container.documents.ingest(
            b"RAG uses external source evidence to support generated answers.", "paper.txt"
        )
        record = container.runs.submit(
            ResearchRequest(question="Explain RAG evidence", max_rounds=1)
        )
        await container.runs.tasks[record.id]
        finished = container.repository.run(record.id)
        assert finished.status == "failed"
        assert "1,000/1,000" in finished.error
        failed = next(
            e for e in finished.events if e.agent == "summarizer" and e.status == "failed"
        )
        assert failed.details["session_input_tokens"] == 990
        assert failed.details["session_output_tokens"] == 10
        assert failed.details["session_calls"] == 1
    finally:
        await container.close()
