"""Plan once; retrieve/rerank per aspect; interleave so one topic cannot take all slots."""

from research_assist.domain.errors import ProviderError
from research_assist.domain.models import (
    QuestionAspect,
    QuestionPlan,
    ResearchRequest,
    RetrievalPlan,
    RetrievedChunk,
)
from research_assist.domain.planning import PLAN_INSTRUCTION, local_plan, needs_planner
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

    async def prepare(self, request: ResearchRequest) -> QuestionPlan:
        fallback = local_plan(request)
        if not needs_planner(request):
            return fallback
        plan = await self.llm.generate(
            task="question_plan",
            instruction=PLAN_INSTRUCTION,
            payload={"question": request.question, "requirements": request.requirements},
            schema=QuestionPlan,
            fallback=lambda: fallback,
        )
        if request.requirements:
            # An LLM may translate queries, but cannot silently change user requirements.
            queries = {a.text: a.query for a in plan.aspects}
            plan = QuestionPlan(
                aspects=[
                    QuestionAspect(text=t, query=queries.get(t, t)) for t in request.requirements
                ]
            )
        return plan

    async def run(
        self,
        request: ResearchRequest,
        gaps: list[str],
        question_plan: QuestionPlan | None = None,
        pending_ids: set[str] | None = None,
        seen_chunk_ids: set[str] | None = None,
    ) -> tuple[RetrievalPlan, list[RetrievedChunk], list[str]]:
        question_plan = question_plan or await self.prepare(request)
        aspects = question_plan.aspects
        focused = [
            a for i, a in enumerate(aspects, 1) if pending_ids is None or f"rq_{i}" in pending_ids
        ]
        # A retry uses missing aspects first, with original intent retained as a separate lane.
        queries = list(dict.fromkeys([a.query for a in focused] + [request.question]))[:13]
        warnings: list[str] = []
        if len(queries) > self.candidate_limit:
            queries = [*queries[: self.candidate_limit - 1], request.question]
            warnings.append("Ngân sách candidate thấp hơn số ý; một số truy vấn chờ vòng bổ sung.")
        plan = RetrievalPlan(queries=queries, facets=[a.text for a in aspects])
        candidates: list[list[RetrievedChunk]] = []
        seen_chunk_ids = seen_chunk_ids or set()
        for query in queries:
            chunks, notes = await self.search.search(
                [query],
                request.document_ids,
                self.candidate_limit + len(seen_chunk_ids),
                request.retrieval_mode,
            )
            warnings.extend(notes)
            # Explore deeper ranked hits on a retry; repeating the same top_k cannot
            # repair a missing answer. Previously accepted evidence stays in the ledger.
            if seen_chunk_ids:
                fresh = [c for c in chunks if c.id not in seen_chunk_ids]
                previous = [c for c in chunks if c.id in seen_chunk_ids]
                chunks = [*fresh, *previous][: self.candidate_limit]
            candidates.append(chunks)
        # Share the expensive reranking budget across distinct chunks. Looking deeper in each
        # local search avoids losing slots when many aspect queries return the same first hits.
        lanes: list[list[RetrievedChunk]] = [[] for _ in queries]
        allocated: set[str] = set()
        for rank in range(max((len(lane) for lane in candidates), default=0)):
            for index, lane in enumerate(candidates):
                if rank < len(lane) and lane[rank].id not in allocated:
                    lanes[index].append(lane[rank])
                    allocated.add(lane[rank].id)
                    if len(allocated) >= self.candidate_limit:
                        break
            if len(allocated) >= self.candidate_limit:
                break
        for index, (query, chunks) in enumerate(zip(queries, lanes, strict=True)):
            before = [c.id for c in chunks]
            rerank_failed = False
            try:
                if chunks:
                    chunks = await self.reranker.rank(query, chunks)
            except ProviderError as exc:
                rerank_failed = True
                if not self.reranker_fallback:
                    raise
                warnings.append(f"Reranker không khả dụng; dùng thứ tự hybrid/BM25. {exc}")
            if seen_chunk_ids:
                chunks.sort(key=lambda c: c.id in seen_chunk_ids)
            plan.trace.append(
                {
                    "query": query,
                    "previously_seen_count": len(seen_chunk_ids),
                    "reranker": {
                        "provider": getattr(self.reranker, "provider", "local"),
                        "model": getattr(self.reranker, "model_name", None),
                        **getattr(self.reranker, "last_rank", {}),
                        "fallback": rerank_failed,
                    },
                    "before_rerank": before,
                    "after_rerank": [
                        {
                            "id": c.id,
                            "page": c.page,
                            "score": c.score,
                            "rerank_score": c.rerank_score,
                        }
                        for c in chunks
                    ],
                }
            )
            lanes[index] = chunks
        selected: list[RetrievedChunk] = []
        seen: set[str] = set()
        for rank in range(max((len(lane) for lane in lanes), default=0)):
            for lane in lanes:
                if rank < len(lane) and lane[rank].id not in seen:
                    selected.append(lane[rank])
                    seen.add(lane[rank].id)
                    if len(selected) >= request.top_k:
                        break
            if len(selected) >= request.top_k:
                break
        if len(focused) > request.top_k:
            warnings.append("Số ý cần trả lời vượt top_k; tăng số đoạn hoặc dùng vòng bổ sung.")
        expand = getattr(self.search, "expand_context", None)
        if expand:
            selected = await expand(selected, request.document_ids)
        return plan, selected, list(dict.fromkeys(warnings))
