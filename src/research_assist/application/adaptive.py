"""Adaptive feedback: typed work requests, autonomous search and selective revision."""

from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar

from research_assist.agents.fact_checker import FactCheckerAgent
from research_assist.agents.reader import ReaderAgent
from research_assist.agents.retrieval import RetrievalAgent
from research_assist.agents.summarizer import SummarizerAgent
from research_assist.application.research_state import ResearchState
from research_assist.domain.coordination import make_task, selected_tasks, work_requests
from research_assist.domain.coverage import gap_messages, reconcile_gaps, requirements_for
from research_assist.domain.models import (
    AgentTask,
    Claim,
    CoordinationDecision,
    Evidence,
    FollowUpRequest,
    QuestionPlan,
    ResearchRequest,
    RetrievedChunk,
)

T = TypeVar("T")
Event = Callable[..., None]


class Step(Protocol):
    def __call__(
        self,
        agent: str,
        message: str,
        work: Awaitable[T],
        describe: Callable[[T], dict[str, object]] | None = None,
    ) -> Awaitable[T]: ...


class AdaptiveCoordinator:
    def __init__(
        self,
        retrieval: RetrievalAgent,
        reader: ReaderAgent,
        summarizer: SummarizerAgent,
        checker: FactCheckerAgent,
        max_steps: int,
    ) -> None:
        self.retrieval, self.reader = retrieval, reader
        self.summarizer, self.checker = summarizer, checker
        self.max_steps = max_steps

    async def run(
        self,
        request: ResearchRequest,
        planned: ResearchRequest,
        plan: QuestionPlan,
        state: ResearchState,
        step: Step,
        event: Event,
    ) -> None:
        # Bootstrap one evidence pass; subsequent work is selected from actual feedback.
        count = await self._round(request, planned, plan, state, step, event)
        if not count:
            state.stop_reason = "no_retrieval_results"
            return
        attempted: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
        for _ in range(self.max_steps):
            if state.coverage and all(c.status == "covered" for c in state.coverage):
                state.stop_reason = "requirements_covered"
                return
            for item in work_requests(state.requests, state.coverage, requirements_for(planned)):
                task = make_task(item)
                if item not in state.requests:
                    task.requested_by = "orchestrator"
                state.tasks.setdefault(task.id, task)
            pending = {c.requirement_id for c in state.coverage if c.status != "covered"}
            claim_ids = {c.id for c in state.claims}
            available = [
                t
                for t in state.tasks.values()
                if t.status in {"pending", "unresolved"}
                and pending.intersection(t.requirement_ids)
                and set(t.claim_ids).issubset(claim_ids)
                and (t.action != "retrieve" or state.rounds < request.max_rounds)
            ]
            if not available:
                state.stop_reason = (
                    "round_limit" if state.rounds >= request.max_rounds else "no_tasks"
                )
                return
            action = available[0].action
            decision = CoordinationDecision(
                action=action,
                task_ids=[t.id for t in available if t.action == action][:4],
                reason="Xử lý yêu cầu bổ sung hợp lệ theo thứ tự được ghi nhận.",
            )
            chosen = selected_tasks(decision, available)
            if chosen is None:
                state.stop_reason = "invalid_agent_action"
                state.warnings.append("Nhiệm vụ không hợp lệ; đã dừng và giữ kết quả kiểm chứng.")
                return
            if not chosen:
                state.stop_reason = "tasks_completed"
                return
            signature = (
                decision.action,
                tuple(sorted(t.id for t in chosen)),
                tuple(sorted(e.id for e in state.evidence))
                + tuple(sorted(c.id for c in state.claims)),
            )
            if signature in attempted:
                state.stop_reason = "no_progress"
                return
            attempted.add(signature)
            for task in chosen:
                task.status, task.attempts = "running", task.attempts + 1
            event(
                "orchestrator", "running", decision.reason, tasks=[t.model_dump() for t in chosen]
            )
            if decision.action == "retrieve":
                state.rounds += 1
            before = {e.id for e in state.evidence}
            await self._round(
                request,
                planned,
                plan,
                state,
                step,
                event,
                tasks=chosen,
                action=decision.action,
                pending={rid for t in chosen for rid in t.requirement_ids},
            )
            covered = {c.requirement_id for c in state.coverage if c.status == "covered"}
            for task in state.tasks.values():
                if set(task.requirement_ids).issubset(covered):
                    task.status = "resolved"
                elif task in chosen:
                    task.status = "unresolved"
            state.history.append(
                {
                    "action": decision.action,
                    "task_ids": decision.task_ids,
                    "new_evidence": len({e.id for e in state.evidence} - before),
                    "covered_requirement_ids": sorted(covered),
                }
            )
            event(
                "orchestrator",
                "completed",
                "Đã đối chiếu kết quả nhiệm vụ.",
                tasks=[t.model_dump() for t in chosen],
                observation=state.history[-1],
            )
        state.stop_reason = (
            "requirements_covered"
            if all(c.status == "covered" for c in state.coverage)
            else "step_limit"
        )

    async def _round(
        self,
        request: ResearchRequest,
        planned: ResearchRequest,
        plan: QuestionPlan,
        state: ResearchState,
        step: Step,
        event: Event,
        *,
        pending: set[str] | None = None,
        tasks: list[AgentTask] | None = None,
        action: str = "retrieve",
    ) -> int:
        requirements = requirements_for(planned)
        requests: list[FollowUpRequest] = list(tasks or [])
        chunks: list[RetrievedChunk] = []
        new_evidence: list[Evidence] = []
        reader_gaps: list[str] = []
        if action == "retrieve":
            retrieval_plan, chunks, notes = await step(
                "retrieval",
                "Tự chọn truy vấn và tìm bằng chứng cho nhiệm vụ.",
                self.retrieval.investigate(
                    request,
                    plan,
                    requests,
                    pending,
                    set(state.retrieved),
                    state.history,
                ),
                lambda v: {
                    "queries": v[0].queries,
                    "stage": "agent_search",
                    "retrieval_lanes": v[0].trace,
                    "facets": v[0].facets,
                    "retrieved_count": len(v[1]),
                    "candidates": [c.model_dump(exclude={"text"}) for c in v[1]],
                    "warnings": v[2],
                },
            )
            state.warnings.extend(notes)
            state.history.append(
                {
                    "action": "search",
                    "queries": retrieval_plan.queries,
                    "retrieved_chunk_ids": [c.id for c in chunks],
                }
            )
        elif action == "read_context":
            requested = {cid for t in (tasks or []) for cid in t.chunk_ids}
            chunks = [c for cid, c in state.retrieved.items() if cid in requested]
            expand = getattr(self.retrieval.search, "expand_context", None)
            if expand:
                chunks = await step(
                    "reader",
                    "Mở rộng ngữ cảnh nguồn đã chọn.",
                    expand(chunks, request.document_ids),
                )
        state.retrieved.update({c.id: c for c in chunks})
        if action in {"retrieve", "read_context"}:
            new_evidence, reader_gaps, notes = await step(
                "reader",
                f"Đọc {len(chunks)} đoạn và kiểm tra trích dẫn nguyên văn.",
                self.reader.run(request.question, chunks, planned.requirements, requests),
                lambda v: {
                    "accepted_evidence": len(v[0]),
                    "evidence": [e.model_dump(exclude={"quote"}) for e in v[0]],
                    "warnings": v[2],
                    "candidate_gaps_unverified": v[1],
                },
            )
            state.warnings.extend(notes)

        # Candidate work remains local until checking and coverage reconciliation finish.
        ledger = {(e.document_id, e.page, e.quote): e for e in state.evidence}
        previous_keys = set(ledger)
        for item in new_evidence:
            key = (item.document_id, item.page, item.quote)
            old = ledger.get(key)
            ledger[key] = (
                old.model_copy(
                    update={
                        "requirement_ids": list(
                            dict.fromkeys(old.requirement_ids + item.requirement_ids)
                        ),
                    }
                )
                if old
                else item
            )
        ordered = [e for k, e in ledger.items() if k not in previous_keys]
        ordered += [e for k, e in ledger.items() if k in previous_keys]
        focused = [r for r in requirements if pending is None or r.id in pending]
        reserved = [
            e for r in focused for e in [e for e in ordered if r.id in e.requirement_ids][:2]
        ]
        pool = list({e.id: e for e in [*reserved, *ordered]}.values())[:24]
        supported = {c.claim_id for c in state.checks if c.verdict == "supported"}
        retained = [c for c in state.claims if c.id in supported]
        synthesis_gaps: list[str] = []
        if action == "verify":
            claims = list(state.claims)
        else:
            candidates, synthesis_gaps, notes = await step(
                "summarizer",
                f"Tổng hợp {len(pool)} bằng chứng cho các ý cần xử lý.",
                self.summarizer.run(request.question, pool, request.language, focused, requests),
            )
            state.warnings.extend(notes)
            targeted = {cid for t in (tasks or []) for cid in t.claim_ids}
            retained = [c for c in retained if c.id not in targeted]
            # Keep a slot for each unaffected aspect before spending slots on new claims.
            protected = [
                c
                for r in requirements
                if pending is not None and r.id not in pending
                for c in [c for c in retained if r.id in c.requirement_ids][:1]
            ]
            merged: dict[str, Claim] = {}
            for c in [*protected, *candidates, *retained]:
                previous_claim = merged.get(c.id)
                merged[c.id] = (
                    c.model_copy(
                        update={
                            "requirement_ids": list(
                                dict.fromkeys(previous_claim.requirement_ids + c.requirement_ids)
                            ),
                        }
                    )
                    if previous_claim
                    else c
                )
            claims = list(merged.values())[:16]
        # A verified claim keeps its proof even if synthesis selects a different 24-item pool.
        pinned = {eid for c in claims for eid in c.evidence_ids}
        evidence = list(
            {
                e.id: e
                for e in [
                    *[e for e in ledger.values() if e.id in pinned],
                    *pool,
                ]
            }.values()
        )
        candidate_gaps = list(dict.fromkeys([*reader_gaps, *synthesis_gaps]))[:16]
        verification = await step(
            "fact_checker",
            f"Kiểm tra mức hỗ trợ và tính nhất quán của {len(claims)} nhận định.",
            self.checker.assess(request.question, claims, evidence, requirements, candidate_gaps),
            lambda v: {
                "checks": [c.model_dump() for c in v.checks],
                "coverage": [c.model_dump() for c in v.coverage],
                "gap_checks": [c.model_dump() for c in v.gap_checks],
                "requests": [r.model_dump() for r in v.requests],
            },
        )
        coverage, gap_details = reconcile_gaps(
            requirements,
            verification.coverage,
            candidate_gaps,
            verification.gap_checks,
            evidence,
        )
        state.evidence, state.claims, state.checks = evidence, claims, verification.checks
        state.coverage, state.gap_details = coverage, gap_details
        state.gaps = gap_messages(gap_details, requirements, request.language)
        await state.save_checkpoint()
        state.requests = verification.requests
        event(
            "orchestrator",
            "completed",
            "Đối chiếu độ đầy đủ và khoảng trống.",
            coverage=[c.model_dump() for c in coverage],
            gaps=[g.model_dump() for g in gap_details],
        )
        return len(chunks)
