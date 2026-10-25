"""Regression cases for translated dates, factual guards and report coverage."""

import pytest

from research_assist.agents.fact_checker import FactCheckerAgent
from research_assist.agents.policies.verification import numeric_guard_reason, validate_checks
from research_assist.agents.writer import WriterAgent
from research_assist.domain.models import (
    Claim,
    ClaimCheck,
    CoverageCheck,
    FactCheckOutput,
    Requirement,
    ResearchRequest,
)
from research_assist.infrastructure.llm import DemoLanguageModel


@pytest.mark.parametrize(
    "source,claim",
    [
        ("11 April 2014", "ngày 11 tháng 4 năm 2014"),
        ("April 11, 2014", "ngày 11 tháng 4 năm 2014"),
        ("11 APR. 2014", "2014-04-11"),
        ("April 11th, 2014", "11 April 2014"),
        ("ngày 11 tháng 4 năm 2014", "April 11, 2014"),
        ("2014-04-11", "ngày 11 tháng 4 năm 2014"),
        ("29 February 2024", "ngày 29 tháng 2 năm 2024"),
        ("11 September 2014", "11 Sept. 2014"),
        ("not 11 April 2014", "không phải ngày 11 tháng 4 năm 2014"),
        ("Published on 11 April 2014.", "Published in 2014."),
    ],
)
def test_equivalent_explicit_dates_pass(source, claim):
    assert numeric_guard_reason(claim, [source]) == ""


@pytest.mark.parametrize(
    "source,claim",
    [
        ("11 April 2014", "ngày 4 tháng 11 năm 2014"),
        ("11 April 2014, revised 12 May 2015", "ngày 12 tháng 4 năm 2014"),
        ("11 April 2014, revised 12 May 2015", "ngày 11 tháng 4 năm 2015"),
        ("11 April 2014", "ngày 31 tháng 4 năm 2014"),
        ("29 February 2024", "ngày 29 tháng 2 năm 2023"),
        ("April 2014. There were 11 authors.", "ngày 11 tháng 4 năm 2014"),
        ("11 April 2014", "Published on 11 April 2014 with 4 authors."),
        ("11 April 2014", "Published on 11 April 2014 with 99% accuracy."),
        ("11 April 2014", "not 11 April 2014"),
        ("not 11 April 2014", "ngày 11 tháng 4 năm 2014"),
        ("11 April 2014", "2014-11-04"),
        ("11 April 2014", "-11 April 2014"),
        ("11 April 2014", "11 April 2014%"),
        ("2014-04-11", "-2014-04-11"),
        ("04/11/2014", "ngày 11 tháng 4 năm 2014"),
        ("11 April 2014. Accuracy is 5%.", "Published on 11 April 2014 at -5%."),
        ("11 April 2014. The sample is not 500.", "Published on 11 April 2014 with 500."),
    ],
)
def test_wrong_or_unsupported_dates_and_measurements_stay_rejected(source, claim):
    assert numeric_guard_reason(claim, [source])


def test_date_cannot_be_assembled_across_citations():
    assert numeric_guard_reason("ngày 11 tháng 4 năm 2014", ["11 April", "2014"])


@pytest.mark.parametrize("verdict", ["insufficient", "contradicted"])
def test_equivalent_date_never_overrides_semantic_rejection(evidence, verdict):
    source = evidence.model_copy(update={"quote": "The first draft appeared on 11 April 2014."})
    claim = Claim(
        id="cl_date",
        text="Bài báo chính thức xuất bản ngày 11 tháng 4 năm 2014.",
        evidence_ids=[source.id],
    )
    check = ClaimCheck(
        claim_id=claim.id,
        verdict=verdict,
        evidence_ids=[source.id],
        explanation="Draft date does not establish final publication date.",
    )
    assert validate_checks([claim], [source], [check])[0].verdict == verdict


async def test_translated_publication_date_reaches_report_with_full_coverage(evidence):
    # Mirrors the failed real run, including the footer and DOI surrounding the date.
    source = evidence.model_copy(
        update={
            "quote": "Digital Object Identifier 10.1109/MCI.2014.2307227\n"
            "Date of publication:11 April 2014\n"
            "48 IEEE COMPUTATIONAL INTELLIGENCE MAGAZINE | MAY 2014",
        }
    )
    requirement = Requirement(id="rq_date", text="Bài báo xuất bản khi nào?")
    claim = Claim(
        id="cl_date",
        text="Bài báo được xuất bản vào ngày 11 tháng 4 năm 2014.",
        evidence_ids=[source.id],
        requirement_ids=[requirement.id],
    )
    check = ClaimCheck(
        claim_id=claim.id,
        verdict="supported",
        evidence_ids=[source.id],
        explanation="The source explicitly states the publication date.",
    )
    checker = FactCheckerAgent(DemoLanguageModel())
    checked = checker.validate_output(
        FactCheckOutput(
            checks=[check],
            coverage=[
                CoverageCheck(
                    requirement_id=requirement.id,
                    status="covered",
                    claim_ids=[claim.id],
                    explanation="Publication date answered.",
                )
            ],
        ),
        [claim],
        [source],
        [requirement],
    )
    assert checked.checks[0].verdict == "supported"
    assert checked.coverage[0].status == "covered"
    report = await WriterAgent(DemoLanguageModel()).compose(
        ResearchRequest(question=requirement.text, requirements=[requirement.text]),
        [claim],
        checked.checks,
        [source],
        [],
        deterministic=True,
    )
    assert claim.id in report.claim_ids
    assert claim.text in report.markdown and source.chunk_id in report.markdown
