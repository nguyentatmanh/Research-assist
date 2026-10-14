import time
from collections.abc import Awaitable, Callable
from typing import Literal, TypeVar

from research_assist.agents.fact_checker import FactCheckerAgent
from research_assist.agents.reader import ReaderAgent
from research_assist.agents.retrieval import RetrievalAgent
from research_assist.agents.summarizer import SummarizerAgent
from research_assist.agents.writer import WriterAgent
from research_assist.domain.models import (
    AgentEvent,
    Claim,
    ClaimCheck,
    Evidence,
    ResearchRequest,
    ResearchResult,
    RetrievedChunk,
)
from research_assist.domain.ports import LanguageModel
from research_assist.infrastructure.llm import current_usage, usage_scope
from research_assist.infrastructure.repository import now

T = TypeVar("T")
EventSink = Callable[[AgentEvent], None]


class ResearchPipeline:
    def __init__(
        self,
        llm: LanguageModel,
        retrieval: RetrievalAgent,
        reader: ReaderAgent,
        summarizer: SummarizerAgent,
        checker: FactCheckerAgent,
        writer: WriterAgent,
        embedding_model: str | None,
        profile: str = "academic",
    ) -> None:
        self.llm, self.retrieval, self.reader = llm, retrieval, reader
        self.summarizer, self.checker, self.writer = summarizer, checker, writer
        self.embedding_model = embedding_model
        self.profile = profile

    async def run(self, run_id: str, request: ResearchRequest, emit: EventSink) -> ResearchResult:
        started = time.perf_counter()
        events: list[AgentEvent] = []
        all_chunks: dict[str, RetrievedChunk] = {}
        all_evidence: dict[tuple[str, int | None, str], Evidence] = {}
        warnings: list[str] = []
        gaps: list[str] = []
        claims: list[Claim] = []
        checks: list[ClaimCheck] = []
        round_number = 1

        def event(
            agent: str,
            status: Literal["running", "completed", "failed", "cancelled"],
            message: str,
            elapsed: float = 0,
            **details: object,
        ) -> None:
            item = AgentEvent(
                sequence=len(events) + 1,
                agent=agent,
                status=status,
                round=round_number,
                message=message,
                elapsed_ms=round(elapsed, 2),
                details=details,
                created_at=now(),
            )
            events.append(item)
            emit(item)

        async def step(agent: str, message: str, work: Awaitable[T]) -> T:
            event(agent, "running", message)
            begin = time.perf_counter()
            usage = current_usage.get()
            previous_ms = usage.agents[agent].elapsed_ms if usage and agent in usage.agents else 0.0
            try:
                value = await work
            except Exception:
                event(
                    agent,
                    "failed",
                    "Bước này không hoàn thành.",
                    (time.perf_counter() - begin) * 1000,
                )
                raise
            elapsed = (time.perf_counter() - begin) * 1000
            details: dict[str, object] = {}
            if usage and agent in usage.agents:
                entry = usage.agents[agent]
                entry.elapsed_ms = round(previous_ms + elapsed, 2)
                details = entry.model_dump(exclude={"agent", "elapsed_ms"})
            event(agent, "completed", "Hoàn thành.", elapsed, **details)
            return value

        with usage_scope(request.profile or self.profile, request.use_cache) as usage:
            for round_number in range(1, request.max_rounds + 1):
                plan, chunks, retrieval_warnings = await step(
                    "retrieval",
                    "Lập kế hoạch và truy xuất tài liệu.",
                    self.retrieval.run(request, gaps),
                )
                warnings.extend(retrieval_warnings)
                all_chunks.update({chunk.id: chunk for chunk in chunks})
                evidence, reader_gaps, reader_warnings = await step(
                    "reader",
                    f"Đọc {len(chunks)} đoạn và kiểm tra trích dẫn nguyên văn.",
                    self.reader.run(request.question, chunks),
                )
                before = len(all_evidence)
                for item in evidence:
                    all_evidence.setdefault((item.document_id, item.page, item.quote), item)
                warnings.extend(reader_warnings)
                pooled = list(all_evidence.values())[:24]
                # A second search with no new evidence cannot improve the synthesis.
                if round_number > 1 and len(all_evidence) == before:
                    gaps = list(dict.fromkeys([*gaps, *reader_gaps]))[:8]
                    event(
                        "orchestrator",
                        "completed",
                        "Không có bằng chứng mới; kết thúc vòng truy xuất bổ sung.",
                    )
                    break
                claims, synthesis_gaps, synthesis_warnings = await step(
                    "summarizer",
                    f"Tổng hợp {len(pooled)} bằng chứng thành nhận định.",
                    self.summarizer.run(request.question, pooled, request.language),
                )
                warnings.extend(synthesis_warnings)
                checks = await step(
                    "fact_checker",
                    f"Kiểm tra mức hỗ trợ của {len(claims)} nhận định.",
                    self.checker.run(request.question, claims, pooled),
                )
                rejected = [check for check in checks if check.verdict != "supported"]
                gaps = list(dict.fromkeys([*reader_gaps, *synthesis_gaps]))[:8]
                if not gaps and not rejected and any(c.verdict == "supported" for c in checks):
                    break
                if rejected:
                    # Use rejected claim text as a bounded follow-up hint.
                    missing = {c.id: c.text for c in claims}
                    gaps = list(dict.fromkeys([*gaps, *[missing[c.claim_id] for c in rejected]]))[
                        :8
                    ]
                if not chunks:
                    break
            final_evidence = list(all_evidence.values())[:24]
            title, markdown, writer_warnings = await step(
                "writer",
                "Biên soạn báo cáo từ nhận định đã được nguồn hỗ trợ.",
                self.writer.run(request, claims, checks, final_evidence, gaps),
            )
            warnings.extend(writer_warnings)
            if usage.missing_usage:
                warnings.append(
                    "Một số phản hồi thiếu thống kê token; số token hiển thị chưa đầy đủ."
                )
            if self.llm.name == "demo":
                warnings.append(
                    "Demo trích xuất: không gọi LLM; "
                    "kết quả không thể dùng để đánh giá chất lượng AI."
                )
            result = ResearchResult(
                run_id=run_id,
                question=request.question,
                title=title,
                markdown=markdown,
                retrieved=list(all_chunks.values()),
                evidence=final_evidence,
                claims=claims,
                checks=checks,
                gaps=gaps,
                events=events,
                provider=self.llm.name,
                model=self.llm.model,
                embedding_model=self.embedding_model,
                llm_calls=usage.calls,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                rounds=round_number,
                warnings=list(dict.fromkeys(warnings)),
                created_at=now(),
                profile=request.profile or self.profile,
                cache_hits=usage.cache_hits,
                agent_usage=list(usage.agents.values()),
                estimated_cost_usd=(
                    sum(e.estimated_cost_usd or 0 for e in usage.agents.values())
                    if usage.agents
                    and all(e.estimated_cost_usd is not None for e in usage.agents.values())
                    else None
                ),
            )
        return result
