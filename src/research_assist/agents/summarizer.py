from hashlib import sha256

from research_assist.domain.models import Claim, Evidence, RawClaim, SynthesisOutput
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
    ) -> tuple[list[Claim], list[str], list[str]]:
        if not evidence:
            return [], [], []
        output = await self.llm.generate(
            task="synthesize_claims",
            instruction=(
                "Synthesize 4–12 atomic, question-relevant claims in the requested language. "
                "Each claim must be directly entailed by the cited evidence_ids, and must "
                "preserve uncertainty and scope. A claim is one factual assertion, not a "
                "heading, instruction, speculation or recommendation. Compare sources "
                "only if both have evidence. Never add results, percentages or references. "
                "Do not put citation markers in the text. Return evidence gaps separately."
            ),
            payload={
                "question": question,
                "language": language,
                "evidence": [e.model_dump() for e in evidence],
            },
            schema=SynthesisOutput,
            fallback=lambda: SynthesisOutput(
                claims=[RawClaim(text=e.quote, evidence_ids=[e.id]) for e in evidence[:12]], gaps=[]
            ),
        )
        valid_ids = {e.id for e in evidence}
        claims, rejected, seen = [], 0, set()
        for raw in output.claims:
            if any(key not in valid_ids for key in raw.evidence_ids):
                rejected += 1
                continue
            if raw.text in seen:
                continue
            seen.add(raw.text)
            claims.append(
                Claim(
                    id="cl_"
                    + sha256(
                        (raw.text + "|" + "|".join(sorted(raw.evidence_ids))).encode()
                    ).hexdigest()[:24],
                    text=raw.text,
                    evidence_ids=list(dict.fromkeys(raw.evidence_ids)),
                )
            )
        warnings = (
            [f"Summarizer: loại {rejected} nhận định dẫn tới evidence_id không tồn tại."]
            if rejected
            else []
        )
        return claims, output.gaps, warnings
