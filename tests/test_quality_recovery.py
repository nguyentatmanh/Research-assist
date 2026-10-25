import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from research_assist.bootstrap import build_container
from research_assist.domain.models import ResearchRequest
from tests.test_autonomous import AutonomousProvider, run_autonomous


async def test_late_writer_outage_keeps_verified_report(settings):
    provider = AutonomousProvider()

    def handle(request):
        task = json.loads(request.content)["response_format"]["json_schema"]["name"]
        return httpx.Response(503) if task == "writer_step" else provider(request)

    settings.llm_retry_initial_seconds = 0.1
    result = await run_autonomous(settings, handle)
    assert result.stop_reason == "provider_error"
    assert result.reported_claim_ids and provider.original_claim in result.markdown
    assert all(c.verdict == "supported" for c in result.checks)
    assert any("503" in warning for warning in result.warnings)
    assert any(s.role == "writer" and s.status == "blocked" for s in result.agent_sessions)


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_checkpoint_persisted_before_writer_and_survives_interruption(settings, stop):
    provider = AutonomousProvider()
    entered = asyncio.Event()

    async def handle(request):
        task = json.loads(request.content)["response_format"]["json_schema"]["name"]
        if task == "writer_step":
            entered.set()
            await asyncio.Event().wait()
        return provider(request)

    settings.llm_provider = "openai"
    settings.openai_api_key = SecretStr("offline-only-key")
    settings.orchestration_mode = "autonomous"
    settings.llm_requests_per_minute = 1000
    settings.run_timeout_seconds = 1
    container = build_container(settings, httpx.MockTransport(handle))
    try:
        doc, _ = await container.documents.ingest(
            b"RAG uses retrieved source evidence to support generated answers.", "paper.txt"
        )
        run = container.runs.submit(
            ResearchRequest(
                question="Explain RAG evidence",
                document_ids=[doc.id],
                scope_mode="selected",
                use_cache=False,
            )
        )
        await asyncio.wait_for(entered.wait(), 2)
        saved = container.repository.run(run.id)
        assert saved.status == "running" and saved.result.reported_claim_ids
        assert provider.original_claim in saved.result.markdown
        if stop == "cancel":
            final = await container.runs.cancel(run.id)
            assert final.status == "cancelled" and final.result.stop_reason == "cancelled"
        else:
            await container.runs.tasks[run.id]
            final = container.repository.run(run.id)
            assert final.status == "completed" and final.result.stop_reason == "timeout"
        assert final.result.reported_claim_ids == saved.result.reported_claim_ids
        assert final.result.markdown == saved.result.markdown
        assert final.result.warnings
    finally:
        await container.close()
