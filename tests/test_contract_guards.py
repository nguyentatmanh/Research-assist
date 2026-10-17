import json

import httpx
import pytest
from pydantic import ValidationError

from research_assist.agents.fact_checker import FactCheckerAgent
from research_assist.agents.writer import WriterAgent
from research_assist.config import Settings
from research_assist.domain.coverage import report_coverage, requirements_for, validate_coverage
from research_assist.domain.json_output import parse_object
from research_assist.domain.models import (
    Claim,
    ClaimCheck,
    CoverageCheck,
    Evidence,
    ResearchRequest,
)
from research_assist.domain.workflow import decide_next
from research_assist.infrastructure.llm import HTTPLanguageModel
from tests.conftest import ScriptedLLM


def claim_and_check(evidence: Evidence) -> tuple[Claim, ClaimCheck]:
    claim = Claim(id="cl_good", text=evidence.quote, evidence_ids=[evidence.id])
    check = ClaimCheck(
        claim_id=claim.id,
        verdict="supported",
        explanation="Source supports it.",
        evidence_ids=[evidence.id],
    )
    return claim, check


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"x":{"a":1,"a":2}}', '{"a":NaN}', "[]"])
def test_ambiguous_json_is_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        parse_object(text)


@pytest.mark.parametrize("value", ["false", 0, None])
def test_request_booleans_are_not_coerced(value: object) -> None:
    with pytest.raises(ValidationError):
        ResearchRequest.model_validate({"question": "Explain RAG methods", "use_cache": value})


async def test_writer_rejects_duplicate_support_at_own_boundary(evidence: Evidence) -> None:
    claim, check = claim_and_check(evidence)
    llm = ScriptedLLM({})
    report = await WriterAgent(llm).compose(
        ResearchRequest(question="Explain RAG methods"), [claim], [check, check], [evidence], []
    )
    assert not report.claim_ids and not llm.calls


async def test_writer_rejects_foreign_evidence_at_own_boundary(evidence: Evidence) -> None:
    claim, check = claim_and_check(evidence)
    check.evidence_ids = ["ev_foreign"]
    report = await WriterAgent(ScriptedLLM({})).compose(
        ResearchRequest(question="Explain RAG methods"), [claim], [check], [evidence], []
    )
    assert not report.claim_ids


async def test_writer_title_and_heading_cannot_smuggle_facts(evidence: Evidence) -> None:
    claim, check = claim_and_check(evidence)
    llm = ScriptedLLM(
        {
            "write_report": {
                "title": "RAG cures cancer with 100% success",
                "sections": [{"heading": "Accuracy is 99%", "blocks": [{"claim_id": claim.id}]}],
            }
        }
    )
    report = await WriterAgent(llm).compose(
        ResearchRequest(question="Explain RAG methods"), [claim], [check], [evidence], []
    )
    assert "cancer" not in report.markdown and "99%" not in report.markdown
    assert report.claim_ids == [claim.id] and report.warnings


async def test_negated_number_does_not_become_positive_support(evidence: Evidence) -> None:
    evidence = evidence.model_copy(update={"quote": "The study used 50 documents, not 500."})
    claim = Claim(id="cl_bad", text="The study used 500 documents.", evidence_ids=[evidence.id])
    llm = ScriptedLLM(
        {
            "check_claims": {
                "checks": [
                    {
                        "claim_id": claim.id,
                        "verdict": "supported",
                        "explanation": "500 is in the text",
                        "evidence_ids": [evidence.id],
                    }
                ]
            }
        }
    )
    checks = await FactCheckerAgent(llm).run("How many documents?", [claim], [evidence])
    assert checks[0].verdict == "insufficient"


async def test_exact_negation_quote_remains_supported(evidence: Evidence) -> None:
    evidence = evidence.model_copy(update={"quote": "The study used 50 documents, not 500."})
    claim, check = claim_and_check(evidence)
    llm = ScriptedLLM({"check_claims": {"checks": [check.model_dump()]}})
    checks = await FactCheckerAgent(llm).run("How many documents?", [claim], [evidence])
    assert checks[0].verdict == "supported"


@pytest.mark.parametrize("kind", ["missing", "duplicate", "foreign_claim", "empty_claims"])
def test_bad_coverage_cannot_be_covered(evidence: Evidence, kind: str) -> None:
    claim, check = claim_and_check(evidence)
    reqs = requirements_for(ResearchRequest(question="Explain RAG methods"))
    row = CoverageCheck(
        requirement_id="rq_1", status="covered", claim_ids=[claim.id], explanation="Fully answered"
    )
    rows = [row]
    if kind == "missing":
        rows = []
    elif kind == "duplicate":
        rows = [row, row]
    elif kind == "foreign_claim":
        row.claim_ids = ["unknown"]
    else:
        row.claim_ids = []
    assert validate_coverage(reqs, rows, [check])[0].status == "unassessed"


def test_writer_omission_invalidates_coverage() -> None:
    row = CoverageCheck(
        requirement_id="rq_1", status="covered", claim_ids=["a", "b"], explanation="Both parts"
    )
    checked = report_coverage([row], ["a"])[0]
    assert checked.status == "unassessed" and checked.claim_ids == ["a"]


def test_missing_aspect_requests_new_evidence_and_stops_at_limit(evidence: Evidence) -> None:
    _, check = claim_and_check(evidence)
    row = CoverageCheck(
        requirement_id="rq_1", status="missing", explanation="No evidence for limitations"
    )
    kwargs = dict(
        max_rounds=2, chunk_count=4, new_evidence_count=2, checks=[check], coverage=[row], gaps=[]
    )
    assert decide_next(round_number=1, **kwargs).retrieve_again
    assert decide_next(round_number=2, **kwargs).reason == "round_limit"


async def test_coverage_added_without_an_extra_model_call(evidence: Evidence) -> None:
    claim, check = claim_and_check(evidence)
    llm = ScriptedLLM(
        {
            "check_claims": {
                "checks": [check.model_dump()],
                "coverage": [
                    {
                        "requirement_id": "rq_1",
                        "status": "covered",
                        "claim_ids": [claim.id],
                        "explanation": "Answers the requirement",
                    }
                ],
            }
        }
    )
    result = await FactCheckerAgent(llm).assess(
        "Explain RAG methods",
        [claim],
        [evidence],
        requirements_for(ResearchRequest(question="Explain RAG methods")),
    )
    assert result.coverage[0].status == "covered" and llm.calls == ["check_claims"]


async def test_duplicate_provider_keys_trigger_one_repair(settings: Settings) -> None:
    from research_assist.domain.models import RetrievalPlan

    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        text = (
            '{"queries":["A"],"queries":["B"]}'
            if len(calls) == 1
            else json.dumps({"queries": ["valid query"]})
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    settings.llm_provider = "openai"
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await HTTPLanguageModel(settings, client).generate(
            task="retrieval_plan",
            instruction="Plan",
            payload={"question": "RAG"},
            schema=RetrievalPlan,
            fallback=lambda: RetrievalPlan(queries=["fallback"]),
        )
    assert result.queries == ["valid query"] and len(calls) == 2


async def test_oversized_prompt_is_rejected_before_any_http_request(settings: Settings) -> None:
    from research_assist.domain.errors import ProviderError
    from research_assist.domain.models import RetrievalPlan

    settings.llm_provider = "openai"
    settings.llm_max_input_chars = 2000
    calls = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: calls.append(r))
    ) as client:
        with pytest.raises(ProviderError, match="MAX_INPUT_CHARS"):
            await HTTPLanguageModel(settings, client).generate(
                task="retrieval_plan",
                instruction="Plan",
                payload={"question": "x" * 3000},
                schema=RetrievalPlan,
                fallback=lambda: RetrievalPlan(queries=["fallback"]),
            )
    assert not calls
