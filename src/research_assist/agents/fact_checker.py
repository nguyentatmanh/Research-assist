from research_assist.agents.policies.verification import numbers, validate_checks  # noqa: F401
from research_assist.agents.prompts.fact_checker import INSTRUCTION
from research_assist.domain.coverage import validate_coverage
from research_assist.domain.models import (
    Claim,
    ClaimCheck,
    Evidence,
    FactCheckOutput,
    Requirement,
)
from research_assist.domain.ports import LanguageModel


class FactCheckerAgent:
    name = "fact_checker"

    def __init__(self, llm: LanguageModel) -> None:
        self.llm = llm

    async def run(
        self, question: str, claims: list[Claim], evidence: list[Evidence]
    ) -> list[ClaimCheck]:
        """Compatibility entry point for integrations that only need claim checks."""
        return (await self.assess(question, claims, evidence, [])).checks

    async def assess(
        self,
        question: str,
        claims: list[Claim],
        evidence: list[Evidence],
        requirements: list[Requirement],
    ) -> FactCheckOutput:
        if not claims:
            return FactCheckOutput(coverage=validate_coverage(requirements, [], []))
        output = await self.llm.generate(
            task="check_claims",
            instruction=INSTRUCTION,
            payload={
                "question": question,
                "claims": [c.model_dump() for c in claims],
                "evidence": [e.model_dump() for e in evidence],
                "requirements": [r.model_dump() for r in requirements],
            },
            schema=FactCheckOutput,
            fallback=lambda: self._extractive_checks(claims, evidence),
        )
        checks = validate_checks(claims, evidence, output.checks)
        return FactCheckOutput(
            checks=checks, coverage=validate_coverage(requirements, output.coverage, checks)
        )

    def _extractive_checks(self, claims: list[Claim], evidence: list[Evidence]) -> FactCheckOutput:
        lookup = {e.id: e for e in evidence}
        checks = []
        for claim in claims:
            sources = [lookup[key] for key in claim.evidence_ids if key in lookup]
            exact = any(claim.text == source.quote for source in sources)
            checks.append(
                ClaimCheck(
                    claim_id=claim.id,
                    verdict="supported" if exact else "insufficient",
                    evidence_ids=[s.id for s in sources],
                    explanation=(
                        "Demo: nhận định là nguyên văn bằng chứng; chưa đánh giá ngữ nghĩa."
                        if exact
                        else "Demo chỉ xác nhận nhận định trích xuất nguyên văn."
                    ),
                )
            )
        # No semantic coverage is asserted by an extractive demo.
        return FactCheckOutput(checks=checks)
