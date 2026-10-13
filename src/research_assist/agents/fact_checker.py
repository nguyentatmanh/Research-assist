import re

from research_assist.domain.models import Claim, ClaimCheck, Evidence, FactCheckOutput
from research_assist.domain.ports import LanguageModel


def numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)*(?:%|％)?", text))


class FactCheckerAgent:
    name = "fact_checker"

    def __init__(self, llm: LanguageModel) -> None:
        self.llm = llm

    async def run(
        self,
        question: str,
        claims: list[Claim],
        evidence: list[Evidence],
    ) -> list[ClaimCheck]:
        if not claims:
            return []
        output = await self.llm.generate(
            task="check_claims",
            instruction=(
                "Act as a conservative evidence checker. For EACH claim, judge semantic "
                "entailment using only its cited quotes, and relevance to the question. "
                "supported: all parts, scope and numbers are directly backed. contradicted: "
                "a cited quote explicitly disagrees. insufficient: missing support, unrelated, "
                "speculative, exaggerated or ambiguous. Shared keywords alone are NOT "
                "entailment. Return one check per claim_id, with a short reason and the "
                "evidence_ids actually used (subset of that claim's evidence_ids). "
                "A citation's existence is NOT proof of a claim. Do not use outside knowledge."
            ),
            payload={
                "question": question,
                "claims": [c.model_dump() for c in claims],
                "evidence": [e.model_dump() for e in evidence],
            },
            schema=FactCheckOutput,
            fallback=lambda: self._extractive_checks(claims, evidence),
        )
        lookup = {e.id: e for e in evidence}
        groups: dict[str, list[ClaimCheck]] = {}
        for check in output.checks:
            groups.setdefault(check.claim_id, []).append(check)
        validated = []
        for claim in claims:
            checks = groups.get(claim.id, [])
            if len(checks) != 1:
                validated.append(
                    ClaimCheck(
                        claim_id=claim.id,
                        verdict="insufficient",
                        evidence_ids=[],
                        explanation="Thiếu hoặc trùng kết quả kiểm tra cho nhận định này.",
                    )
                )
                continue
            check = checks[0]
            if not check.evidence_ids or any(
                key not in lookup or key not in claim.evidence_ids for key in check.evidence_ids
            ):
                check = ClaimCheck(
                    claim_id=claim.id,
                    verdict="insufficient",
                    evidence_ids=[],
                    explanation="Kết quả kiểm tra không trỏ tới bằng chứng hợp lệ của nhận định.",
                )
            elif check.verdict == "supported":
                quote_text = " ".join(lookup[key].quote for key in check.evidence_ids)
                if not numbers(claim.text).issubset(numbers(quote_text)):
                    check = ClaimCheck(
                        claim_id=claim.id,
                        verdict="insufficient",
                        evidence_ids=check.evidence_ids,
                        explanation="Có con số không xuất hiện trong các trích dẫn đã kiểm tra.",
                    )
            validated.append(check)
        return validated

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
                        "Demo: nhận định là nguyên văn bằng chứng; "
                        "chưa có kiểm tra entailment bằng LLM."
                        if exact
                        else "Demo chỉ xác nhận nhận định trích xuất nguyên văn."
                    ),
                )
            )
        return FactCheckOutput(checks=checks)
