from research_assist.domain.models import ResearchRequest, RetrievalPlan, RetrievedChunk
from research_assist.domain.ports import LanguageModel, Reranker, SearchTool


class RetrievalAgent:
    name = "retrieval"

    def __init__(
        self, llm: LanguageModel, search: SearchTool, reranker: Reranker, candidate_limit: int = 20
    ) -> None:
        self.llm, self.search, self.reranker = llm, search, reranker
        self.candidate_limit = candidate_limit

    async def run(
        self,
        request: ResearchRequest,
        gaps: list[str],
    ) -> tuple[RetrievalPlan, list[RetrievedChunk], list[str]]:
        plan = await self.llm.generate(
            task="retrieval_plan",
            instruction=(
                "Produce 1–3 concise search queries for the question and any evidence gaps. "
                "Keep the original intent; add an English query if the question is Vietnamese. "
                "Use only search terms, not answers. Do not invent named papers or results. "
                "facets are aspects that the sources should cover."
            ),
            payload={"question": request.question, "evidence_gaps": gaps, "style": request.style},
            schema=RetrievalPlan,
            fallback=lambda: RetrievalPlan(queries=[request.question, *gaps[:2]], facets=[]),
        )
        # The original question always participates, so a poor rewrite cannot erase it.
        plan.queries = list(dict.fromkeys([request.question, *plan.queries]))[:3]
        chunks, warnings = await self.search.search(
            plan.queries,
            request.document_ids,
            max(request.top_k, min(request.top_k * 3, self.candidate_limit)),
            request.retrieval_mode,
        )
        chunks = await self.reranker.rank(request.question, chunks)
        return plan, chunks[: request.top_k], warnings
