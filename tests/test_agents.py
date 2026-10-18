import pytest

from research_assist.agents.fact_checker import FactCheckerAgent
from research_assist.agents.reader import ReaderAgent, quote_span
from research_assist.agents.summarizer import SummarizerAgent
from research_assist.agents.writer import WriterAgent
from research_assist.domain.models import (
    Claim,
    ClaimCheck,
    Evidence,
    ResearchRequest,
    RetrievedChunk,
)
from research_assist.infrastructure.llm import DemoLanguageModel
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
