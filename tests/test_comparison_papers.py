import json
import re

import fitz
import httpx
import pytest
from pydantic import SecretStr

from research_assist.agents.fact_checker import FactCheckerAgent
from research_assist.agents.retrieval import RetrievalAgent
from research_assist.bootstrap import build_container
from research_assist.domain.models import (
    Claim,
    ClaimCheck,
    CoverageCheck,
    FactCheckOutput,
    Requirement,
    ResearchRequest,
)
from research_assist.infrastructure.llm import DemoLanguageModel
from research_assist.infrastructure.search import NoReranker
from tests.conftest import ScriptedLLM

QUESTION = "Hai bài báo nghiên cứu bài toán nào? Có thể so trực tiếp BLEU và độ chính xác không?"


def paper_pdf(*pages: str) -> bytes:
    with fitz.open() as pdf:
        for text in pages:
            page = pdf.new_page()
            assert page.insert_textbox(fitz.Rect(50, 50, 540, 300), text, fontsize=12) >= 0
        return pdf.tobytes()


@pytest.mark.parametrize("comparison", [False, True], ids=["bleu", "cross-paper"])
async def test_paper_questions_from_frozen_scope_to_audited_report(settings, comparison):
    if comparison:
        question = (
            "Hai bài báo nghiên cứu bài toán nào? Có thể đối chiếu trực tiếp "
            "điểm BLEU với độ chính xác của bộ phân tích cú pháp không?"
        )
        requirements = [
            "Bài toán và thước đo dịch máy",
            "Bài toán và thước đo phân tích cú pháp",
            "Có thể so trực tiếp BLEU và độ chính xác không?",
        ]
        queries = [
            "machine translation Transformer English-German BLEU",
            "Vietnamese syntactic parsing accuracy",
            "Transformer English-German BLEU Vietnamese syntactic parsing accuracy",
        ]
        source_marks = ["28.4 BLEU", "42.5%"]
    else:
        question = (
            "Transformer big đạt BLEU bao nhiêu trên WMT 2014 English–German và English–French?"
        )
        requirements = ["Điểm BLEU English–German", "Điểm BLEU English–French"]
        queries = ["Transformer big English-German BLEU", "Transformer big English-French BLEU"]
        source_marks = ["28.4 BLEU", "41.0 BLEU"]
    synthesis = (
        "Không thể so sánh trực tiếp các số liệu vì hai bài đánh giá nhiệm vụ khác nhau "
        "bằng BLEU và độ chính xác phân tích cú pháp."
    )

    def provider(http_request):
        body = json.loads(http_request.content)
        payload = json.loads(body["messages"][1]["content"])
        task = body["response_format"]["json_schema"]["name"]
        if task == "question_plan":
            ids = {source["filename"]: source["id"] for source in payload["sources"]}
            papers = [["transformer.pdf"], ["parser.pdf"], ["transformer.pdf", "parser.pdf"]]
            output = {
                "aspects": [
                    {
                        "text": text,
                        "query": query,
                        "document_ids": [ids[p] for p in papers[i]] if comparison else [],
                    }
                    for i, (text, query) in enumerate(zip(requirements, queries, strict=True))
                ]
            }
        elif task == "read_evidence":
            index = requirements.index(payload["requirements"][0])
            shared = comparison and index == 2
            if comparison:
                expected = {"transformer.pdf", "parser.pdf"}
                if not shared:
                    expected = {("transformer.pdf", "parser.pdf")[index]}
                assert {s["filename"] for s in payload["sources"]} == expected
            marks = source_marks if shared else [source_marks[index]]
            units = [
                next(unit for unit in payload["source_units"] if mark in unit["text"])
                for mark in marks
            ]
            output = {
                "evidence": [
                    {
                        "chunk_id": unit["chunk_id"],
                        "unit_id": unit["unit_id"],
                        "quote": "",
                        "relevance": "Published metric.",
                    }
                    for unit in units
                ],
            }
        elif task == "synthesize_claims":
            sources = payload["evidence"]
            claims = [
                {
                    "text": source["quote"],
                    "evidence_ids": [source["id"]],
                    "requirement_ids": source["requirement_ids"],
                }
                for source in sources
            ]
            if comparison:
                claims.append(
                    {
                        "text": synthesis,
                        "evidence_ids": [source["id"] for source in sources],
                        "requirement_ids": ["rq_1", "rq_2", "rq_3"],
                    }
                )
            output = {"claims": claims}
        else:
            assert task == "check_claims"
            output = {
                "checks": [
                    {
                        "claim_id": claim["id"],
                        "verdict": "supported",
                        "evidence_ids": claim["evidence_ids"],
                        "explanation": "Fixture approval.",
                    }
                    for claim in payload["claims"]
                ],
                "coverage": [
                    {
                        "requirement_id": requirement["id"],
                        "status": "covered",
                        "claim_ids": [
                            claim["id"]
                            for claim in payload["claims"]
                            if requirement["id"] in claim["requirement_ids"]
                        ],
                        "explanation": "Covered by the fixture.",
                    }
                    for requirement in payload["requirements"]
                ],
            }
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    settings.llm_provider = "openai"
    settings.openai_api_key = SecretStr("test-only")
    settings.llm_requests_per_minute = 1000
    container = build_container(settings, transport=httpx.MockTransport(provider))
    try:
        transformer, _ = await container.documents.ingest(
            paper_pdf(
                "Transformer big performs machine translation. "
                "It reached 28.4 BLEU on WMT 2014 English-German.",
                "Transformer big performs machine translation. "
                "It reached 41.0 BLEU on WMT 2014 English-French.",
            ),
            "transformer.pdf",
            paper_title="Transformer Research",
        )
        parser, _ = await container.documents.ingest(
            paper_pdf("Vietnamese syntactic parsing used parser accuracy. The accuracy was 42.5%."),
            "parser.pdf",
            paper_title="Vietnamese Parsing",
        )
        other, _ = await container.documents.ingest(
            paper_pdf("This document discusses orbital telemetry and solar navigation."),
            "telemetry.pdf",
            paper_title="Telemetry Notes",
        )
        selected = [transformer.id, parser.id, other.id] if comparison else [transformer.id]
        run = container.runs.submit(
            ResearchRequest(
                question=question,
                requirements=requirements,
                document_ids=selected,
                scope_mode="selected",
                orchestration_mode="autonomous",
                retrieval_mode="lexical",
                top_k=2,
                style="comparison" if comparison else "overview",
                max_rounds=1,
            )
        )
        await container.runs.tasks[run.id]
        record = container.repository.run(run.id)
        assert record.status == "completed", record.error
        result = record.result
        assert result is not None
        assert result.report_status == "complete"
        assert result.citation_audit.status == "passed"
        source_by_marker = {
            str(i): next(e for e in result.evidence if e.id == evidence_id)
            for i, evidence_id in enumerate(result.citation_evidence_ids, 1)
        }
        if comparison:
            readers = [state for state in result.research_tasks if state.evidence_ids]
            assert {frozenset(state.scope.allowed_document_ids) for state in readers} == {
                frozenset({transformer.id}),
                frozenset({parser.id}),
                frozenset({transformer.id, parser.id}),
            }
            header = next(line for line in result.markdown.splitlines() if line.startswith("|"))
            assert "Transformer Research | Vietnamese Parsing | Telemetry Notes |" in header
            assert "| Chưa đủ bằng chứng |" in result.markdown
            cross_claim = next(claim for claim in result.claims if claim.text == synthesis)
            assert cross_claim.id in result.reported_claim_ids
            assert result.markdown.startswith(synthesis)
            cross_line = next(line for line in result.markdown.splitlines() if synthesis in line)
            assert not cross_line.startswith("|")
            assert {
                source_by_marker[marker].document_id
                for marker in re.findall(r"\[(\d+)\]", cross_line)
            } == {transformer.id, parser.id}
            for line in result.markdown.splitlines():
                if line.startswith("|") and "28.4 BLEU" in line:
                    assert {
                        source_by_marker[marker].document_id
                        for marker in re.findall(r"\[(\d+)\]", line)
                    } == {transformer.id}
                    break
            else:
                pytest.fail("The table omitted the supported Transformer metric")
        else:
            for english_pair, value, page in (("German", "28.4", 1), ("French", "41.0", 2)):
                line = next(
                    line for line in result.markdown.splitlines() if value in line and "[" in line
                )
                marker = re.search(r"\[(\d+)\]", line)
                assert marker is not None
                evidence = source_by_marker[marker[1]]
                assert evidence.document_id == transformer.id
                assert evidence.page == page
                assert f"English-{english_pair}" in evidence.quote
    finally:
        await container.close()


async def test_planner_preserves_an_explicit_two_paper_comparison(settings, document):
    other = document.model_copy(update={"id": "second", "filename": "other.pdf"})
    planner = ScriptedLLM(
        {"question_plan": {"aspects": [{"text": "Bài toán", "query": "machine translation"}]}}
    )
    container = build_container(settings)
    try:
        agent = RetrievalAgent(planner, container.search, NoReranker())
        request = ResearchRequest(
            question=QUESTION, scope_mode="selected", document_ids=[document.id, other.id]
        )
        plan = await agent.prepare(request, sources=[document, other])
        assert [a.text for a in plan.aspects] == [
            "Bài toán",
            "Có thể so trực tiếp BLEU và độ chính xác không?",
        ]
        assert plan.aspects[-1].document_ids == request.document_ids
    finally:
        await container.close()


@pytest.mark.parametrize(
    ("scenario", "verdict", "coverage"),
    [
        ("incompatible", "supported", "covered"),
        ("same-benchmark", "supported", "covered"),
        ("bad-ranking", "insufficient", "unassessed"),
        ("missing-metric", "insufficient", "unassessed"),
        ("only-one-paper", "insufficient", "unassessed"),
        ("partial-description", "supported", "partial"),
    ],
)
def test_comparison_requires_an_approved_conclusion_with_evidence_from_both_papers(
    evidence, scenario, verdict, coverage
):
    question = QUESTION
    sources = [
        evidence.model_copy(update={"quote": "Machine translation reached 28.4 BLEU on WMT 2014."}),
        evidence.model_copy(
            update={
                "id": "ev_other",
                "document_id": "doc_other",
                "quote": "Vietnamese syntactic parsing achieved an accuracy of 42.5%.",
            }
        ),
    ]
    text = (
        "Không thể so sánh trực tiếp vì hai bài báo đo dịch máy bằng BLEU "
        "và phân tích cú pháp bằng độ chính xác."
    )
    if scenario == "same-benchmark":
        question = "Can these two papers be directly compared on the same BLEU benchmark?"
        sources[1].quote = "Machine translation reached 29.1 BLEU on the same WMT 2014 test set."
        text = (
            "The results can be directly compared on the WMT 2014 test set, "
            "where 29.1 BLEU is higher than 28.4 BLEU."
        )
    elif scenario == "bad-ranking":
        text = "42.5% accuracy is higher than 28.4 BLEU."
    elif scenario == "missing-metric":
        sources[1].quote = "Vietnamese syntactic parsing was evaluated on 100 sentences."
    elif scenario == "only-one-paper":
        sources = sources[:1]
    elif scenario == "partial-description":
        text = "The papers studied machine translation and Vietnamese syntactic parsing."
    requirement = Requirement(id="rq_1", text=question)
    claim = Claim(
        id="cl_compare",
        text=text,
        evidence_ids=[source.id for source in sources],
        requirement_ids=[requirement.id],
    )
    output = FactCheckOutput(
        checks=[
            ClaimCheck(
                claim_id=claim.id,
                verdict="supported",
                explanation="Fixture attempts approval.",
                evidence_ids=claim.evidence_ids,
            )
        ],
        coverage=[
            CoverageCheck(
                requirement_id=requirement.id,
                status="covered",
                claim_ids=[claim.id],
                explanation="Fixture attempts full coverage.",
            )
        ],
    )
    audited = FactCheckerAgent(DemoLanguageModel()).validate_output(
        output, [claim], sources, [requirement], question=question
    )
    assert audited.checks[0].verdict == verdict
    assert audited.coverage[0].status == coverage
