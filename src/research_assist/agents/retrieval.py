from research_assist.agents.prompts.retrieval import INSTRUCTION
from research_assist.domain.errors import ProviderError
from research_assist.domain.models import ResearchRequest, RetrievalPlan, RetrievedChunk
from research_assist.domain.ports import LanguageModel, Reranker, SearchTool


class RetrievalAgent:
    name = "retrieval"

    def __init__(
        self,
        llm: LanguageModel,
        search: SearchTool,
        reranker: Reranker,
        candidate_limit: int = 20,
        reranker_fallback: bool = True,
    ) -> None:
        self.llm, self.search, self.reranker = llm, search, reranker
        self.candidate_limit = candidate_limit
        self.reranker_fallback = reranker_fallback

    async def run(
        self,
        request: ResearchRequest,
        gaps: list[str],
    ) -> tuple[RetrievalPlan, list[RetrievedChunk], list[str]]:
        plan = await self.llm.generate(
            task="retrieval_plan",
            instruction=(INSTRUCTION),
            payload={
                "question": request.question,
                "evidence_gaps": gaps,
                "style": request.style,
                "requirements": request.requirements,
            },
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
        try:
            chunks = await self.reranker.rank(request.question, chunks)
        except ProviderError:
            if not self.reranker_fallback:
                raise
            warnings.append("Reranker không khả dụng; dùng thứ tự hybrid/BM25 cho phiên này.")
        return plan, chunks[: request.top_k], warnings
