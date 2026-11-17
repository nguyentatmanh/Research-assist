import json

import httpx
import pytest

from research_assist.agents.fact_checker import FactCheckerAgent
from research_assist.agents.reader import ReaderAgent, quote_span
from research_assist.agents.summarizer import SummarizerAgent
from research_assist.agents.writer import WriterAgent
from research_assist.application.methodology import methodology_checklist
from research_assist.domain.models import (
    Claim,
    ClaimCheck,
    Document,
    Evidence,
    RawClaim,
    Requirement,
    ResearchRequest,
    RetrievedChunk,
    SynthesisOutput,
    TaskBudget,
)
from research_assist.infrastructure.llm import (
    DemoLanguageModel,
    HTTPLanguageModel,
    current_task_budget,
    usage_scope,
)
from tests.conftest import ScriptedLLM


@pytest.mark.parametrize(
    "quote,found",
    [
        ("RAG combines retrieval with generation.", True),
        ("RAG combines  retrieval\nwith generation.", True),
        ("RAG achieved 99% accuracy in 2026.", False),
        ("rag combines retrieval with generation.", False),
    ],
)
def test_quotes_must_exist_verbatim(chunk: RetrievedChunk, quote: str, found: bool) -> None:
    assert (quote_span(chunk.text, quote) is not None) == found


async def test_reader_discards_invented_quotes_and_ids(chunk: RetrievedChunk) -> None:
    llm = ScriptedLLM(
        {
            "read_evidence": {
                "evidence": [
                    {
                        "chunk_id": chunk.id,
                        "quote": "RAG combines retrieval with generation.",
                        "relevance": "Relevant",
                    },
                    {
                        "chunk_id": chunk.id,
                        "quote": "RAG achieves 99% accuracy without evidence.",
                        "relevance": "Fake",
                    },
                    {
                        "chunk_id": "ch_fake",
                        "quote": "RAG combines retrieval with generation.",
                        "relevance": "Fake",
                    },
                ],
                "gaps": [],
            }
        }
    )
    evidence, _, warnings = await ReaderAgent(llm).run("Explain RAG methods", [chunk])
    assert len(evidence) == 1
    assert evidence[0].page == 2
    assert evidence[0].start == 10
    assert "2" in warnings[0]


async def test_reader_keeps_focused_evidence_within_actual_utf8_reservation(settings, chunk):
    requested = "BLEU WMT 2014 English–German"
    result = "In WMT 2014 English-to-German translation, the model reached 28.4 BLEU."
    padding = "Nghiên cứu đánh giá nhiều phương pháp khác nhau trong thực nghiệm. " * 40
    texts = [padding] * 10 + [result]
    chunks = [
        chunk.model_copy(
            update={"id": f"ch_{i}", "text": text, "start": i * 4000, "end": i * 4000 + len(text)}
        )
        for i, text in enumerate(texts)
    ]
    settings.llm_provider = "openai"
    settings.llm_max_output_tokens = 2000
    budget = TaskBudget(max_tokens=12000)
    seen = []

    def respond(request):
        body = json.loads(request.content)
        system, user = [message["content"] for message in body["messages"]]
        assert len((system + user).encode()) + body["max_tokens"] <= budget.max_tokens
        windows = json.loads(user)["source_units"]
        assert windows[0]["text"] == result
        seen.extend(windows)
        output = {
            "evidence": [
                {
                    "chunk_id": windows[0]["chunk_id"],
                    "unit_id": windows[0]["unit_id"],
                    "quote": "",
                    "relevance": "Requested evaluation result.",
                }
            ],
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(output)}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        reader = ReaderAgent(HTTPLanguageModel(settings, client))
        with usage_scope() as usage:
            usage.strict_reservations = True
            token = current_task_budget.set(budget)
            try:
                evidence, gaps, warnings = await reader.run(requested, chunks, [requested])
            finally:
                current_task_budget.reset(token)
    assert [item.quote for item in evidence] == [result]
    assert evidence[0].start == chunks[-1].start
    assert seen and len(seen) < len(texts) * 3
    assert not gaps and any("ngân sách" in warning for warning in warnings)


async def test_reader_collects_quoted_metrics_when_no_source_answers_comparison_directly(chunk):
    question = "Có thể so trực tiếp điểm BLEU với độ chính xác phân tích cú pháp không?"
    texts = ["The translation model scored 28.4 BLEU.", "Độchính xác phân tích cú pháp là 42.5%."]
    chunks = [
        chunk.model_copy(
            update={"id": f"ch_{i}", "document_id": f"doc_{i}", "text": text, "end": 10 + len(text)}
        )
        for i, text in enumerate(texts)
    ]
    model = ScriptedLLM({"read_evidence": {"evidence": []}})
    evidence, _, _ = await ReaderAgent(model).run(question, chunks, cover_documents=True)
    assert {item.quote for item in evidence} == set(texts)
    assert {item.document_id for item in evidence} == {c.document_id for c in chunks}


async def test_synthesis_rejects_nonexistent_evidence(evidence: Evidence) -> None:
    llm = ScriptedLLM(
        {
            "synthesize_claims": {
                "claims": [
                    {"text": "RAG uses retrieval and generation.", "evidence_ids": [evidence.id]},
                    {"text": "RAG beats every model.", "evidence_ids": ["ev_fake"]},
                ],
                "gaps": [],
            }
        }
    )
    claims, _, warnings = await SummarizerAgent(llm).run("Explain RAG methods", [evidence], "en")
    assert len(claims) == 1 and warnings


def test_synthesis_dedup_preserves_each_requirement_and_claim_kind(evidence):
    text = "RAG uses retrieval and generation."
    output = SynthesisOutput(
        claims=[
            RawClaim(text=text, evidence_ids=[evidence.id], requirement_ids=["r1"]),
            RawClaim(text=text, evidence_ids=[evidence.id, evidence.id], requirement_ids=["r2"]),
            RawClaim(
                text=text,
                evidence_ids=[evidence.id],
                requirement_ids=["r2"],
                kind="grounded_recommendation",
            ),
        ]
    )
    requirements = [Requirement(id=rid, text=rid) for rid in ["r1", "r2"]]
    claims, _, warnings = SummarizerAgent(ScriptedLLM({})).validate_output(
        output, [evidence], requirements
    )
    assert len(claims) == 2 and not warnings
    assert claims[0].requirement_ids == ["r1", "r2"]
    assert claims[0].evidence_ids == [evidence.id]
    assert claims[0].id != claims[1].id
    assert claims[0].kind == "source_finding" and claims[1].kind == "grounded_recommendation"


def test_synthesis_rejects_ambiguous_ledger_identity(evidence):
    output = SynthesisOutput(claims=[RawClaim(text=evidence.quote, evidence_ids=[evidence.id])])
    claims, _, warnings = SummarizerAgent(ScriptedLLM({})).validate_output(
        output, [evidence, evidence.model_copy(update={"quote": "A conflicting source text."})]
    )
    assert not claims and warnings


@pytest.mark.parametrize(
    "checks",
    [
        [],
        [
            {
                "claim_id": "cl_test",
                "verdict": "supported",
                "explanation": "OK",
                "evidence_ids": ["ev_fake"],
            },
        ],
        [
            {
                "claim_id": "cl_test",
                "verdict": "supported",
                "explanation": "OK",
                "evidence_ids": ["ev_test"],
            },
            {
                "claim_id": "cl_test",
                "verdict": "supported",
                "explanation": "OK",
                "evidence_ids": ["ev_test"],
            },
        ],
    ],
)
async def test_missing_duplicate_or_invalid_checks_fail_closed(
    evidence: Evidence, checks: list[dict[str, object]]
) -> None:
    llm = ScriptedLLM({"check_claims": {"checks": checks}})
    claim = Claim(
        id="cl_test", text="RAG uses retrieval and generation.", evidence_ids=[evidence.id]
    )
    checked = await FactCheckerAgent(llm).run("Explain RAG methods", [claim], [evidence])
    assert checked[0].verdict == "insufficient"


async def test_unbacked_number_cannot_pass_llm_judge(evidence: Evidence) -> None:
    claim = Claim(id="cl_test", text="RAG achieves 99% accuracy.", evidence_ids=[evidence.id])
    llm = ScriptedLLM(
        {
            "check_claims": {
                "checks": [
                    {
                        "claim_id": claim.id,
                        "verdict": "supported",
                        "explanation": "Fake pass",
                        "evidence_ids": [evidence.id],
                    }
                ]
            }
        }
    )
    checks = await FactCheckerAgent(llm).run("Explain RAG results", [claim], [evidence])
    assert checks[0].verdict == "insufficient"
    assert "con số" in checks[0].explanation


async def test_demo_judge_does_not_claim_semantic_entailment(evidence: Evidence) -> None:
    claim = Claim(
        id="cl_test", text="RAG always produces correct answers.", evidence_ids=[evidence.id]
    )
    checked = await FactCheckerAgent(DemoLanguageModel()).run("Explain RAG", [claim], [evidence])
    assert checked[0].verdict == "insufficient"


async def test_checker_and_writer_preserve_supported_translation(evidence: Evidence) -> None:
    # A stub judgement tests the application guards, not real-model semantic accuracy.
    source = evidence.model_copy(
        update={"quote": "The encoder is composed of a stack of 6 identical layers."}
    )
    claim = Claim(
        id="cl_translation",
        text="Encoder gồm 6 lớp giống nhau.",
        evidence_ids=[source.id],
    )
    assert claim.text != source.quote
    llm = ScriptedLLM(
        {
            "check_claims": {
                "checks": [
                    {
                        "claim_id": claim.id,
                        "verdict": "supported",
                        "explanation": "The translation preserves the meaning and count.",
                        "evidence_ids": [source.id],
                    }
                ]
            }
        }
    )
    checks = await FactCheckerAgent(llm).run("Encoder có bao nhiêu lớp?", [claim], [source])
    assert checks[0].verdict == "supported"
    _, markdown, _ = await WriterAgent(DemoLanguageModel()).run(
        ResearchRequest(question="Encoder có bao nhiêu lớp?"),
        [claim],
        checks,
        [source],
        [],
    )
    assert claim.text in markdown and source.chunk_id in markdown


async def test_semantically_rejected_claim_is_not_rescued_by_shared_words(
    evidence: Evidence,
) -> None:
    source = evidence.model_copy(update={"quote": "The encoder has 6 layers."})
    claim = Claim(id="cl_wrong_entity", text="The decoder has 6 layers.", evidence_ids=[source.id])
    llm = ScriptedLLM(
        {
            "check_claims": {
                "checks": [
                    {
                        "claim_id": claim.id,
                        "verdict": "insufficient",
                        "explanation": "The quote describes the encoder, not the decoder.",
                        "evidence_ids": [source.id],
                    }
                ]
            }
        }
    )
    checks = await FactCheckerAgent(llm).run("How many decoder layers?", [claim], [source])
    assert checks[0].verdict == "insufficient"
    report = await WriterAgent(DemoLanguageModel()).compose(
        ResearchRequest(question="How many decoder layers?"), [claim], checks, [source], []
    )
    assert not report.claim_ids


async def test_writer_cannot_emit_unverified_claims(evidence: Evidence) -> None:
    good = Claim(id="cl_good", text=evidence.quote, evidence_ids=[evidence.id])
    bad = Claim(
        id="cl_bad", text="RAG always produces correct answers.", evidence_ids=[evidence.id]
    )
    checks = [
        ClaimCheck(
            claim_id=good.id,
            verdict="supported",
            explanation="Supported",
            evidence_ids=[evidence.id],
        ),
        ClaimCheck(
            claim_id=bad.id,
            verdict="insufficient",
            explanation="Unbacked",
            evidence_ids=[evidence.id],
        ),
    ]
    llm = ScriptedLLM(
        {
            "write_report": {
                "title": "RAG overview",
                "sections": [
                    {
                        "heading": "Evidence",
                        "blocks": [
                            {"claim_id": good.id},
                            {"claim_id": bad.id},
                            {"claim_id": "cl_fake"},
                        ],
                    }
                ],
            }
        }
    )
    _, markdown, warnings = await WriterAgent(llm).run(
        ResearchRequest(question="Explain RAG methods"), [good, bad], checks, [evidence], []
    )
    assert good.text in markdown
    assert bad.text not in markdown
    assert "[1]" in markdown and evidence.chunk_id in markdown
    assert warnings
    # The model only received verified claim IDs, never the rejected factual assertion.
    assert "cl_bad" not in str(llm.payloads[0])


async def test_writer_reports_abstention_without_model_call() -> None:
    llm = ScriptedLLM({})
    _, markdown, _ = await WriterAgent(llm).run(
        ResearchRequest(question="Unanswerable scientific question"), [], [], [], []
    )
    assert "Chưa" in markdown and not llm.calls


async def test_comparison_table_preserves_scope_and_full_cross_document_citations(
    document: Document, evidence: Evidence
) -> None:
    unobserved = document.model_copy(update={"id": "doc_missing", "title": ""})
    first = document.model_copy(update={"title": "Transformer | bài báo"})
    second = document.model_copy(update={"id": "doc_parsing", "title": "Phân tích cú pháp"})
    source_transformer = evidence.model_copy(
        update={"quote": "The Transformer reached 28.4 BLEU on English–German translation."}
    )
    source_parsing = evidence.model_copy(
        update={
            "id": "ev_parsing",
            "chunk_id": "ch_parsing",
            "document_id": second.id,
            "quote": "Bộ phân tích cú pháp tiếng Việt được đánh giá bằng độ chính xác.",
        }
    )
    targeted = ResearchRequest(
        question="Có thể so trực tiếp kết quả của hai bài báo không?",
        document_ids=[unobserved.id, first.id, second.id],
        requirements=["Nhiệm vụ | thước đo", "Cỡ mẫu"],
        style="comparison",
    )
    rows = [
        ("bleu", "Dịch Anh–Đức | đạt 28.4 BLEU.", [source_transformer.id], ["rq_1"]),
        ("parsing", source_parsing.quote, [source_parsing.id], ["rq_1"]),
        (
            "comparison",
            "Khác nhiệm vụ và thước đo, không thể so trực tiếp các kết quả.",
            [source_transformer.id, source_parsing.id],
            ["rq_1"],
        ),
        ("unmapped", "Kết quả bên Transformer là 28.4 BLEU.", [source_transformer.id], ["rq_3"]),
        ("incorrect_bleu", "Bài báo đạt 41.1 BLEU.", [source_transformer.id], ["rq_1"]),
    ]
    claims = [
        Claim(id=key, text=claim_text, evidence_ids=ids, requirement_ids=reqs)
        for key, claim_text, ids, reqs in rows
    ]
    checks = [
        ClaimCheck(
            claim_id=claim.id,
            verdict="supported",
            explanation="App validates the supplied judgement again.",
            evidence_ids=claim.evidence_ids,
        )
        for claim in claims
    ]

    report = await WriterAgent(DemoLanguageModel()).compose(
        targeted,
        claims,
        checks,
        [source_transformer, source_parsing],
        [],
        scope_documents=[unobserved, first, second],
    )

    assert (
        "| Tiêu chí | study.pdf | Transformer \\| bài báo | Phân tích cú pháp |" in report.markdown
    )
    assert (
        "| Nhiệm vụ \\| thước đo | Chưa đủ bằng chứng | "
        "Dịch Anh–Đức \\| đạt 28.4 BLEU. [1] | " + source_parsing.quote + " [2] |"
    ) in report.markdown
    assert "| Cỡ mẫu | Chưa đủ bằng chứng | Chưa đủ bằng chứng | Chưa đủ bằng chứng |" in (
        report.markdown
    )
    assert report.markdown.startswith(f"{claims[2].text} [1] [2]\n\n")
    assert report.markdown.count(claims[2].text) == 1
    assert f"\n{claims[3].text} [1]\n" in report.markdown
    assert "41.1" not in report.markdown
    assert report.claim_ids == ["bleu", "parsing", "comparison", "unmapped"]
    assert report.citation_evidence_ids == [source_transformer.id, source_parsing.id]


@pytest.mark.parametrize(
    ("cited_count", "has_table", "warns"),
    [(0, False, False), (1, False, True), (2, True, False), (6, False, True)],
)
async def test_comparison_broad_scope_uses_only_cited_document_columns(
    document: Document, evidence: Evidence, cited_count: int, has_table: bool, warns: bool
) -> None:
    documents = [
        document.model_copy(update={"id": f"doc_{i}", "title": f"Archived paper {i}"})
        for i in range(6)
    ]
    sources = [
        evidence.model_copy(update={"id": f"ev_{i}", "document_id": doc.id})
        for i, doc in enumerate(documents[:cited_count])
    ]
    claims = [
        Claim(id=f"cl_{i}", text=source.quote, evidence_ids=[source.id], requirement_ids=["rq_1"])
        for i, source in enumerate(sources)
    ]
    checks = [
        ClaimCheck(
            claim_id=claim.id,
            verdict="supported",
            explanation="The quote supports this statement.",
            evidence_ids=claim.evidence_ids,
        )
        for claim in claims
    ]
    report = await WriterAgent(DemoLanguageModel()).compose(
        ResearchRequest(
            question="Compare the document findings.",
            style="comparison",
            language="en",
            document_ids=[doc.id for doc in documents],
        ),
        claims,
        checks,
        sources,
        [],
        scope_documents=documents,
    )
    assert ("## Document comparison" in report.markdown) is has_table
    assert bool(report.warnings) is warns
    assert report.claim_ids == [claim.id for claim in claims]
    if has_table:
        assert "| Criterion | Archived paper 0 | Archived paper 1 |" in report.markdown
        assert "Archived paper 2" not in report.markdown


def test_methodology_checklist_preserves_old_count_form_and_recognizes_sentences(
    evidence: Evidence,
) -> None:
    legacy = evidence.model_copy(update={"quote": "The sample size was N = 100."})
    sentences = evidence.model_copy(update={"id": "ev_sentences", "quote": "Bộ mẫu có 100 câu."})
    result = methodology_checklist([legacy, sentences])
    assert isinstance(result["items"], list)
    sample_size = next(item for item in result["items"] if item["criterion"] == "sample_size")
    assert [item["evidence_id"] for item in sample_size["matches"]] == [legacy.id, sentences.id]
