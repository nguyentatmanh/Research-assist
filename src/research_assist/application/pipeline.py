import time
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol, TypeVar

from research_assist.agents.fact_checker import FactCheckerAgent
from research_assist.agents.reader import ReaderAgent
from research_assist.agents.retrieval import RetrievalAgent
from research_assist.agents.summarizer import SummarizerAgent
from research_assist.agents.writer import WriterAgent, escape_markdown
from research_assist.application.adaptive import AdaptiveCoordinator
from research_assist.application.autonomous import AutonomousCoordinator
from research_assist.application.research_state import ResearchState
from research_assist.domain.coverage import (
    gap_messages,
    reconcile_gaps,
    report_coverage,
    requirements_for,
    validate_coverage,
)
from research_assist.domain.errors import BudgetExceeded, ProviderError
from research_assist.domain.metadata import EVIDENCE_POLICY
from research_assist.domain.models import (
    AgentEvent,
    AgentUsage,
    Evidence,
    EvidenceGap,
    QuestionPlan,
    ResearchRequest,
    ResearchResult,
)
from research_assist.domain.ports import LanguageModel
from research_assist.domain.workflow import decide_next
from research_assist.infrastructure.llm import current_step_events, current_usage, usage_scope
from research_assist.infrastructure.repository import now

T = TypeVar("T")
EventSink = Callable[[AgentEvent], None]
Event = Callable[..., None]


class Step(Protocol):
    def __call__(
        self,
        agent: str,
        message: str,
        work: Awaitable[T],
        describe: Callable[[T], dict[str, object]] | None = None,
    ) -> Awaitable[T]: ...


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
        orchestration_mode: Literal["fixed", "adaptive", "autonomous"] = "autonomous",
        coordinator_max_steps: int = 4,
        agent_max_steps: int = 6,
        autonomous_max_steps: int = 16,
        autonomous_max_tasks: int = 16,
        autonomous_parallelism: int = 5,
        autonomous_team_planning: bool = False,
    ) -> None:
        self.llm, self.retrieval, self.reader = llm, retrieval, reader
        self.summarizer, self.checker, self.writer = summarizer, checker, writer
        self.embedding_model = embedding_model
        self.profile = profile
        self.orchestration_mode = orchestration_mode
        self.coordinator_max_steps = coordinator_max_steps
        self.agent_max_steps = agent_max_steps
        self.autonomous_max_steps = autonomous_max_steps
        self.autonomous_max_tasks = autonomous_max_tasks
        self.autonomous_parallelism = autonomous_parallelism
        self.autonomous_team_planning = autonomous_team_planning

    async def run(
        self,
        run_id: str,
        request: ResearchRequest,
        emit: EventSink,
        on_checkpoint: Callable[[ResearchResult], None] | None = None,
    ) -> ResearchResult:
        started = time.perf_counter()
        state = ResearchState()
        events: list[AgentEvent] = []
        mode = request.orchestration_mode or self.orchestration_mode

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
                round=state.rounds,
                message=message,
                elapsed_ms=round(elapsed, 2),
                details=details,
                created_at=now(),
            )
            events.append(item)
            emit(item)

        async def step(
            agent: str,
            message: str,
            work: Awaitable[T],
            describe: Callable[[T], dict[str, object]] | None = None,
        ) -> T:
            event(agent, "running", message)
            begin = time.perf_counter()
            usage = current_usage.get()
            diagnostics: list[dict[str, object]] = []
            diagnostic_token = current_step_events.set(diagnostics)
            previous_ms = usage.agents[agent].elapsed_ms if usage and agent in usage.agents else 0.0
            try:
                value = await work
            except Exception:
                details: dict[str, object] = {}
                if usage:
                    if diagnostics:
                        details["provider_attempts"] = diagnostics
                    details.update(
                        session_input_tokens=usage.input_tokens,
                        session_output_tokens=usage.output_tokens,
                        session_calls=usage.calls,
                    )
                    if agent in usage.agents:
                        entry = usage.agents[agent]
                        entry.elapsed_ms = round(
                            previous_ms + (time.perf_counter() - begin) * 1000, 2
                        )
                        details.update(entry.model_dump(exclude={"agent", "elapsed_ms"}))
                event(
                    agent,
                    "failed",
                    "Bước này không hoàn thành.",
                    (time.perf_counter() - begin) * 1000,
                    **details,
                )
                raise
            finally:
                current_step_events.reset(diagnostic_token)
            elapsed = (time.perf_counter() - begin) * 1000
            details = {}
            if diagnostics:
                details["provider_attempts"] = diagnostics
            if usage and agent in usage.agents:
                entry = usage.agents[agent]
                entry.elapsed_ms = round(previous_ms + elapsed, 2)
                details.update(entry.model_dump(exclude={"agent", "elapsed_ms"}))
            if describe:
                details.update(describe(value))
            event(agent, "completed", "Hoàn thành.", elapsed, **details)
            return value

        def budget_stop(message: str) -> None:
            state.stop_reason = "budget_exhausted"
            state.warnings.append(
                message + " Báo cáo chỉ giữ nhận định đã kiểm chứng trước khi dừng."
            )
            state.gap_details = [
                EvidenceGap(requirement_id=c.requirement_id, reason="budget_exhausted")
                for c in state.coverage
                if c.status != "covered"
            ]
            state.gaps = gap_messages(
                state.gap_details, requirements_for(planned_request), request.language
            )
            for task in state.tasks.values():
                if task.status == "running":
                    task.status = "unresolved"
            event(
                "orchestrator",
                "completed",
                "Đã dừng ở ngân sách và giữ kết quả đã kiểm chứng.",
                stop_reason=state.stop_reason,
                tasks=[t.model_dump() for t in state.tasks.values()],
            )

        with usage_scope(request.profile or self.profile, request.use_cache, mode) as usage:

            async def checkpoint() -> None:
                if on_checkpoint is None or not any(c.verdict == "supported" for c in state.checks):
                    return
                report = await self.writer.compose(
                    planned_request,
                    state.claims,
                    state.checks,
                    state.evidence,
                    state.gaps,
                    state.gap_details,
                    deterministic=True,
                )
                # Deep copy: subsequent worker mutations must not alter a saved checkpoint.
                saved = ResearchResult(
                    run_id=run_id,
                    question=request.question,
                    title=report.title,
                    markdown=report.markdown,
                    retrieved=list(state.retrieved.values()),
                    evidence=state.evidence,
                    claims=state.claims,
                    checks=state.checks,
                    gaps=state.gaps,
                    events=list(events),
                    provider=self.llm.name,
                    model=self.llm.model,
                    embedding_model=self.embedding_model,
                    llm_calls=usage.calls,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                    rounds=state.rounds,
                    warnings=list(state.warnings),
                    created_at=now(),
                    requirements=requirements_for(planned_request),
                    coverage=report_coverage(state.coverage, report.claim_ids),
                    reported_claim_ids=report.claim_ids,
                    stop_reason="checkpoint",
                    policy_version=EVIDENCE_POLICY,
                    document_ids=request.document_ids,
                    gap_details=state.gap_details,
                    profile=request.profile or self.profile,
                    cache_hits=usage.cache_hits,
                    agent_usage=list(usage.agents.values()),
                    orchestration_mode=mode,
                    task_graph=list(state.task_graph.values()),
                    agent_sessions=list(state.sessions.values()),
                    agent_plans=dict(state.agent_plans),
                    peer_messages=list(state.peer_messages),
                ).model_copy(deep=True)
                on_checkpoint(saved)

            state.checkpoint = checkpoint

            def provider_stop(exc: ProviderError) -> None:
                state.stop_reason = "provider_error"
                state.warnings.append(
                    str(exc) + " Báo cáo giữ phần đã kiểm chứng; bước sau chưa hoàn thành."
                )
                for task in state.task_graph.values():
                    if task.status in {"pending", "running", "waiting"}:
                        task.status, task.message = "blocked", "Provider unavailable."
                for session in state.sessions.values():
                    if session.status in {"pending", "running", "waiting"}:
                        session.status = "blocked"
                event(
                    "orchestrator",
                    "completed",
                    "Giữ kết quả đã kiểm chứng khi API lỗi.",
                    stop_reason=state.stop_reason,
                )

            event(
                "orchestrator",
                "running",
                "Đã cố định phạm vi tài liệu.",
                document_ids=request.document_ids,
                scope_mode=request.scope_mode,
                policy_version=EVIDENCE_POLICY,
                orchestration_mode=mode,
            )
            planned_request = request
            state.coverage = validate_coverage(requirements_for(request), [], [])
            try:
                question_plan = await step(
                    "retrieval",
                    "Phân tích các ý cần trả lời.",
                    self.retrieval.prepare(request),
                    lambda p: {
                        "stage": "question_planning",
                        "aspects": [a.model_dump() for a in p.aspects],
                    },
                )
                planned_request = request.model_copy(
                    update={"requirements": [a.text for a in question_plan.aspects]}
                )
                state.coverage = validate_coverage(requirements_for(planned_request), [], [])
                if mode == "autonomous":
                    autonomous = AutonomousCoordinator(
                        self.llm,
                        self.retrieval,
                        self.reader,
                        self.summarizer,
                        self.checker,
                        self.writer,
                        self.agent_max_steps,
                        self.autonomous_max_steps,
                        self.autonomous_max_tasks,
                        self.autonomous_parallelism,
                        self.autonomous_team_planning,
                    )
                    await autonomous.run(
                        request,
                        planned_request,
                        question_plan,
                        state,
                        step,
                        event,
                    )
                elif mode == "adaptive":
                    coordinator = AdaptiveCoordinator(
                        self.retrieval,
                        self.reader,
                        self.summarizer,
                        self.checker,
                        self.coordinator_max_steps,
                    )
                    await coordinator.run(
                        request, planned_request, question_plan, state, step, event
                    )
                else:
                    await self._fixed(request, planned_request, question_plan, state, step, event)
            except BudgetExceeded as exc:
                budget_stop(str(exc))
            except ProviderError as exc:
                if not any(c.verdict == "supported" for c in state.checks):
                    raise
                provider_stop(exc)
            if mode == "adaptive" or state.stop_reason == "budget_exhausted":
                usage.agents.setdefault(
                    "writer",
                    AgentUsage(
                        agent="writer",
                        provider="local",
                        model="deterministic",
                        execution="local",
                        estimated_cost_usd=0.0,
                    ),
                )
            try:
                report = await step(
                    "writer",
                    "Biên soạn báo cáo từ nhận định đã được nguồn hỗ trợ.",
                    self.writer.compose(
                        planned_request,
                        state.claims,
                        state.checks,
                        state.evidence,
                        state.gaps,
                        state.gap_details,
                        deterministic=mode != "fixed"
                        or state.stop_reason in {"budget_exhausted", "provider_error"},
                        layout=state.report_layout if mode == "autonomous" else None,
                    ),
                    lambda v: {
                        "written_claims": len(v.claim_ids),
                        "stop_reason": state.stop_reason,
                        "stage": "report_rendering",
                    },
                )
            except ProviderError as exc:
                if isinstance(exc, BudgetExceeded):
                    budget_stop(str(exc))
                elif any(c.verdict == "supported" for c in state.checks):
                    provider_stop(exc)
                else:
                    raise
                report = await step(
                    "writer",
                    "Xuất kết quả đã kiểm chứng bằng bộ dựng báo cáo cục bộ.",
                    self.writer.compose(
                        planned_request,
                        state.claims,
                        state.checks,
                        state.evidence,
                        state.gaps,
                        state.gap_details,
                        deterministic=True,
                    ),
                )
            requirements = requirements_for(planned_request)
            warnings, claims, checks = state.warnings, state.claims, state.checks
            coverage, gaps, gap_details = state.coverage, state.gaps, state.gap_details
            final_evidence, all_chunks = state.evidence, state.retrieved
            stop_reason, round_number = state.stop_reason, state.rounds
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
                policy_version=EVIDENCE_POLICY,
                document_ids=request.document_ids,
                gap_details=gap_details,
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
                agent_plans=dict(state.agent_plans),
                peer_messages=list(state.peer_messages),
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
        return result.model_copy(
            update={
                "orchestration_mode": mode,
                "agent_tasks": list(state.tasks.values()),
                "task_graph": list(state.task_graph.values()),
                "agent_sessions": list(state.sessions.values()),
            }
        )

    async def _fixed(
        self,
        request: ResearchRequest,
        planned_request: ResearchRequest,
        question_plan: QuestionPlan,
        state: ResearchState,
        step: Step,
        event: Event,
    ) -> None:
        all_chunks = state.retrieved
        all_evidence: dict[tuple[str, int | None, str], Evidence] = {}
        warnings, claims, checks = state.warnings, state.claims, state.checks
        gaps: list[str] = []
        requirements = requirements_for(planned_request)
        pending_ids = None
        pooled: list[Evidence] = []
        for round_number in range(1, request.max_rounds + 1):
            state.rounds = round_number
            plan, chunks, retrieval_warnings = await step(
                "retrieval",
                "Lập kế hoạch và truy xuất tài liệu.",
                self.retrieval.run(request, gaps, question_plan, pending_ids, set(all_chunks)),
                lambda v: {
                    "queries": v[0].queries,
                    "stage": "local_search",
                    "execution": "local",
                    "retrieval_lanes": v[0].trace,
                    "facets": v[0].facets,
                    "retrieved_count": len(v[1]),
                    "candidates": [
                        {
                            "id": c.id,
                            "document_id": c.document_id,
                            "page": c.page,
                            "section": c.section,
                            "score": c.score,
                            "lexical_score": c.lexical_score,
                            "semantic_score": c.semantic_score,
                            "rerank_score": c.rerank_score,
                        }
                        for c in v[1]
                    ],
                    "warnings": v[2],
                },
            )
            warnings.extend(retrieval_warnings)
            all_chunks.update({chunk.id: chunk for chunk in chunks})
            evidence, reader_gaps, reader_warnings = await step(
                "reader",
                f"Đọc {len(chunks)} đoạn và kiểm tra trích dẫn nguyên văn.",
                self.reader.run(request.question, chunks, planned_request.requirements),
                lambda v: {
                    "accepted_evidence": len(v[0]),
                    "evidence": [
                        {
                            "id": e.id,
                            "chunk_id": e.chunk_id,
                            "unit_id": e.unit_id,
                            "requirement_ids": e.requirement_ids,
                            "start": e.start,
                            "end": e.end,
                        }
                        for e in v[0]
                    ],
                    "warnings": v[2],
                    "candidate_gaps_unverified": v[1],
                },
            )
            previous_keys = set(all_evidence)
            before = len(all_evidence)
            for item in evidence:
                key = (item.document_id, item.page, item.quote)
                previous = all_evidence.get(key)
                all_evidence[key] = (
                    previous.model_copy(
                        update={
                            "requirement_ids": list(
                                dict.fromkeys(
                                    [
                                        *previous.requirement_ids,
                                        *item.requirement_ids,
                                    ]
                                )
                            ),
                        }
                    )
                    if previous
                    else item
                )
            warnings.extend(reader_warnings)
            # Give newly retrieved evidence a chance in round two. Taking the first
            # 24 accumulated items forever would silently discard the follow-up search.
            pooled = [item for key, item in all_evidence.items() if key not in previous_keys]
            pooled += [item for key, item in all_evidence.items() if key in previous_keys]
            # Reserve evidence slots per aspect before filling the remaining budget.
            reserved = []
            for requirement in requirements:
                related = [e for e in pooled if requirement.id in e.requirement_ids]
                reserved.extend(related[:2])
            pooled = list({e.id: e for e in [*reserved, *pooled]}.values())[:24]
            supported_ids = {c.claim_id for c in checks if c.verdict == "supported"}
            previous_claims = [c for c in claims if c.id in supported_ids]
            claims, synthesis_gaps, synthesis_warnings = await step(
                "summarizer",
                f"Tổng hợp {len(pooled)} bằng chứng thành nhận định.",
                self.summarizer.run(request.question, pooled, request.language, requirements),
            )
            warnings.extend(synthesis_warnings)
            # Retain prior findings if their evidence is still in the bounded ledger.
            pool_ids = {e.id for e in pooled}
            current_texts = {c.text for c in claims}
            current_ids = {c.id for c in claims}
            claims.extend(
                c
                for c in previous_claims
                if c.text not in current_texts
                and c.id not in current_ids
                and set(c.evidence_ids).issubset(pool_ids)
            )
            claims = claims[:16]
            candidate_gaps = list(dict.fromkeys([*reader_gaps, *synthesis_gaps]))[:16]
            verification = await step(
                "fact_checker",
                f"Kiểm tra mức hỗ trợ của {len(claims)} nhận định.",
                self.checker.assess(request.question, claims, pooled, requirements, candidate_gaps),
                lambda v: {
                    "checks": [c.model_dump() for c in v.checks],
                    "coverage": [c.model_dump() for c in v.coverage],
                    "gap_checks": [c.model_dump() for c in v.gap_checks],
                },
            )
            checks, coverage = verification.checks, verification.coverage
            coverage, gap_details = reconcile_gaps(
                requirements,
                coverage,
                candidate_gaps,
                verification.gap_checks,
                pooled,
            )
            gaps = gap_messages(gap_details, requirements, request.language)
            pending_ids = {c.requirement_id for c in coverage if c.status != "covered"}
            state.evidence, state.claims, state.checks = pooled, claims, checks
            state.coverage, state.gaps, state.gap_details = coverage, gaps, gap_details
            await state.save_checkpoint()
            event(
                "orchestrator",
                "completed",
                "Đối chiếu độ đầy đủ và khoảng trống.",
                coverage=[c.model_dump() for c in coverage],
                gaps=[g.model_dump() for g in gap_details],
            )
            decision = decide_next(
                round_number=round_number,
                max_rounds=request.max_rounds,
                chunk_count=len(chunks),
                new_evidence_count=len(all_evidence) - before,
                checks=checks,
                coverage=coverage,
                gaps=gaps,
            )
            state.stop_reason = decision.reason
            event(
                "orchestrator",
                "completed",
                decision.reason,
                retrieve_again=decision.retrieve_again,
            )
            if not decision.retrieve_again:
                break
