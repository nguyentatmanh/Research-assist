import json
from collections import Counter
from hashlib import sha256

from research_assist.agents.prompts.summarizer import INSTRUCTION
from research_assist.domain.models import (
    Claim,
    Evidence,
    FollowUpRequest,
    RawClaim,
    Requirement,
    SynthesisOutput,
)
from research_assist.domain.ports import LanguageModel


class SummarizerAgent:
    name = "summarizer"

    def __init__(self, llm: LanguageModel) -> None:
        self.llm = llm

    async def run(
        self,
        question: str,
        evidence: list[Evidence],
        language: str,
        requirements: list[Requirement] | None = None,
        requests: list[FollowUpRequest] | None = None,
    ) -> tuple[list[Claim], list[str], list[str]]:
        if not evidence:
            return [], [], []
        output = await self.llm.generate(
            task="synthesize_claims",
            instruction=(INSTRUCTION),
            payload={
                "question": question,
                "language": language,
                "requirements": [r.model_dump() for r in (requirements or [])],
                "evidence": [e.model_dump(exclude={"regions"}) for e in evidence],
                "revision_requests": [r.model_dump() for r in (requests or [])],
            },
            schema=SynthesisOutput,
            fallback=lambda: SynthesisOutput(
                claims=[
                    RawClaim(text=e.quote, evidence_ids=[e.id])
                    for e in evidence[:12]
                    if len(e.quote) <= 1000
                ],
                gaps=[],
            ),
        )
        return self.validate_output(output, evidence, requirements)

    def validate_output(
        self,
        output: SynthesisOutput,
        evidence: list[Evidence],
        requirements: list[Requirement] | None = None,
    ) -> tuple[list[Claim], list[str], list[str]]:
        counts = Counter(e.id for e in evidence)
        valid_ids = {key for key, count in counts.items() if count == 1}
        requirement_ids = {r.id for r in (requirements or [])}
        claims: list[Claim] = []
        rejected = 0
        seen: dict[tuple[str, str, tuple[str, ...]], Claim] = {}
        for raw in output.claims:
            if any(key not in valid_ids for key in raw.evidence_ids):
                rejected += 1
                continue
            references = list(dict.fromkeys(raw.evidence_ids))
            related = list(
                dict.fromkeys(rid for rid in raw.requirement_ids if rid in requirement_ids)
            )
            identity = (raw.text, raw.kind, tuple(sorted(references)))
            if identity in seen:
                previous = seen[identity]
                previous.requirement_ids = list(
                    dict.fromkeys([*previous.requirement_ids, *related])
                )
                continue
            claim = Claim(
                id="cl_"
                + sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:24],
                text=raw.text,
                evidence_ids=references,
                requirement_ids=related,
                kind=raw.kind,
            )
            claims.append(claim)
            seen[identity] = claim
        warnings = (
            [f"Summarizer: loại {rejected} nhận định có evidence_id thiếu hoặc không duy nhất."]
            if rejected
            else []
        )
        return claims, output.gaps, warnings
