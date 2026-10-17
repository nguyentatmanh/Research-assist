import time
from collections.abc import Awaitable, Callable
from typing import Literal, TypeVar

from research_assist.agents.fact_checker import FactCheckerAgent
from research_assist.agents.reader import ReaderAgent
from research_assist.agents.retrieval import RetrievalAgent
from research_assist.agents.summarizer import SummarizerAgent
from research_assist.agents.writer import WriterAgent, escape_markdown
from research_assist.domain.coverage import report_coverage, requirements_for
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
from research_assist.domain.workflow import decide_next
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
        requirements = requirements_for(request)
        coverage = []
        stop_reason = "round_limit"

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
                events[-1].details.update({"queries": plan.queries, "retrieved_count": len(chunks)})
                all_chunks.update({chunk.id: chunk for chunk in chunks})
                evidence, reader_gaps, reader_warnings = await step(
                    "reader",
                    f"Đọc {len(chunks)} đoạn và kiểm tra trích dẫn nguyên văn.",
                    self.reader.run(request.question, chunks),
                )
                events[-1].details.update(
                    {"accepted_evidence": len(evidence), "warnings": reader_warnings}
                )
                previous_keys = set(all_evidence)
                before = len(all_evidence)
                for item in evidence:
                    all_evidence.setdefault((item.document_id, item.page, item.quote), item)
                warnings.extend(reader_warnings)
                # Give newly retrieved evidence a chance in round two. Taking the first
                # 24 accumulated items forever would silently discard the follow-up search.
                pooled = [item for key, item in all_evidence.items() if key not in previous_keys]
                pooled += [item for key, item in all_evidence.items() if key in previous_keys]
                pooled = pooled[:24]
                # A second search with no new evidence cannot improve the synthesis.
                if round_number > 1 and len(all_evidence) == before:
                    stop_reason = "no_new_evidence"
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
                    self.summarizer.run(request.question, pooled, request.language, requirements),
                )
                warnings.extend(synthesis_warnings)
                verification = await step(
                    "fact_checker",
                    f"Kiểm tra mức hỗ trợ của {len(claims)} nhận định.",
                    self.checker.assess(request.question, claims, pooled, requirements),
                )
                checks, coverage = verification.checks, verification.coverage
                events[-1].details.update(
                    {
                        "supported": sum(c.verdict == "supported" for c in checks),
                        "checked": len(checks),
                        "coverage": [c.model_dump() for c in coverage],
                    }
                )
                rejected = [check for check in checks if check.verdict != "supported"]
                gaps = list(dict.fromkeys([*reader_gaps, *synthesis_gaps]))[:8]
                if rejected:
                    # Use rejected claim text as a bounded follow-up hint.
                    missing = {c.id: c.text for c in claims}
                    gaps = list(dict.fromkeys([*gaps, *[missing[c.claim_id] for c in rejected]]))[
                        :8
                    ]
                unresolved_ids = {
                    c.requirement_id for c in coverage if c.status in {"missing", "partial"}
                }
                gaps = list(
                    dict.fromkeys(
                        [*gaps, *[r.text for r in requirements if r.id in unresolved_ids]]
                    )
                )[:8]
                decision = decide_next(
                    round_number=round_number,
                    max_rounds=request.max_rounds,
                    chunk_count=len(chunks),
                    new_evidence_count=len(all_evidence) - before,
                    checks=checks,
                    coverage=coverage,
                    gaps=gaps,
                )
                stop_reason = decision.reason
                event(
                    "orchestrator",
                    "completed",
                    decision.reason,
                    retrieve_again=decision.retrieve_again,
                )
                if not decision.retrieve_again:
                    break
            final_evidence = pooled
            report = await step(
                "writer",
                "Biên soạn báo cáo từ nhận định đã được nguồn hỗ trợ.",
                self.writer.compose(request, claims, checks, final_evidence, gaps),
            )
            warnings.extend(report.warnings)
            coverage = report_coverage(coverage, report.claim_ids)
            en = request.language == "en"
            labels = {
                "covered": "Đủ ý theo đánh giá model",
                "partial": "Mới đáp ứng một phần",
                "missing": "Thiếu bằng chứng",
                "unassessed": "Chưa đánh giá đầy đủ",
            }
            texts = {r.id: r.text for r in requirements}
            coverage_lines = [
                "",
                "## " + ("Question coverage" if en else "Mức đáp ứng câu hỏi"),
                "",
            ]
            for row in coverage:
                label = row.status if en else labels[row.status]
                coverage_lines.append(
                    "- " + escape_markdown(texts[row.requirement_id]) + ": " + label
                )
            if any(c.status != "covered" for c in coverage):
                warnings.append(
                    "Báo cáo chưa được xác nhận đáp ứng đầy đủ mọi yêu cầu của câu hỏi."
                )
            coverage_lines += [
                "",
                "*Coverage is a model assessment, not independently verified accuracy.*"
                if en
                else (
                    "*Độ đầy đủ do model đánh giá; không phải độ chính xác được xác minh độc lập.*"
                ),
            ]
            markdown = report.markdown + "\n".join(coverage_lines) + "\n"
            events[-1].details.update(
                {"written_claims": len(report.claim_ids), "stop_reason": stop_reason}
            )
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
                requirements=requirements,
                coverage=coverage,
                reported_claim_ids=report.claim_ids,
                stop_reason=stop_reason,
                policy_version="2.1",
                title=report.title,
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
