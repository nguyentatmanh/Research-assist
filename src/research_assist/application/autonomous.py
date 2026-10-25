"""Code-scheduled task graph for five autonomous workers with peer dependencies."""

import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel

from research_assist.agents.autonomous import INSTRUCTIONS
from research_assist.agents.fact_checker import FactCheckerAgent
from research_assist.agents.policies.verification import numbers, numeric_guard_reason
from research_assist.agents.reader import ReaderAgent
from research_assist.agents.retrieval import RetrievalAgent
from research_assist.agents.summarizer import SummarizerAgent
from research_assist.agents.writer import WriterAgent
from research_assist.application.agent_runtime import AgentRuntime, AgentTool, Event, Step
from research_assist.application.research_state import ResearchState
from research_assist.application.task_graph import TaskGraph
from research_assist.domain.coverage import gap_messages, reconcile_gaps, requirements_for
from research_assist.domain.errors import (
    ActionRejected,
    BudgetExceeded,
    ProviderError,
    ProviderUnavailable,
)
from research_assist.domain.models import (
    AgentDecision,
    AgentSession,
    Claim,
    Evidence,
    FactCheckOutput,
    QuestionPlan,
    RawClaim,
    ReaderOutput,
    ResearchRequest,
    RetrievedChunk,
    RuntimeTask,
    SearchCompletion,
    SynthesisOutput,
    TaskRequest,
    TaskRequester,
    ToolArguments,
    WorkerRole,
    WriterOutput,
)
from research_assist.domain.ports import LanguageModel
from research_assist.domain.source_units import source_units
from research_assist.infrastructure.llm import current_usage

T = TypeVar("T", bound=BaseModel)


def merge_source_candidates(previous: list[str], incoming: list[str], limit: int = 12) -> list[str]:
    """Reserve room for earlier ranked hits while exploring a later search round."""
    merged: list[str] = []
    for rank in range(max(len(previous), len(incoming))):
        for lane in (previous, incoming):
            if rank < len(lane) and lane[rank] not in merged:
                merged.append(lane[rank])
                if len(merged) >= limit:
                    return merged
    return merged


class AutonomousCoordinator:
    def __init__(
        self,
        llm: LanguageModel,
        retrieval: RetrievalAgent,
        reader: ReaderAgent,
        summarizer: SummarizerAgent,
        checker: FactCheckerAgent,
        writer: WriterAgent,
        worker_steps: int,
        coordinator_steps: int,
        max_tasks: int,
    ) -> None:
        self.llm, self.retrieval, self.reader = llm, retrieval, reader
        self.summarizer, self.checker, self.writer = summarizer, checker, writer
        self.worker_steps, self.coordinator_steps, self.max_tasks = (
            worker_steps,
            coordinator_steps,
            max_tasks,
        )

    async def run(
        self,
        request: ResearchRequest,
        planned: ResearchRequest,
        plan: QuestionPlan,
        state: ResearchState,
        step: Step,
        event: Event,
    ) -> None:
        # A coordinator belongs to one run; nothing here is shared between concurrent runs.
        self.request, self.planned, self.plan, self.state = request, planned, plan, state
        self.step, self.event = step, event
        self.graph = TaskGraph(state.task_graph, self.max_tasks)
        self.evidence: list[Evidence] = []
        self.proposals: list[Claim] = []
        self.candidate_gaps: list[str] = []
        self.active_chunks: list[str] = []
        self.inspected: dict[str, list[str]] = {}
        self.searches = 0
        self.source_revision = 0
        self.read_revision = -1
        self.draft_revision = -1
        self.artifact_revision = 0
        self.checked_revision = -1
        self.guard_feedback_revision = -1
        self.coverage_repair_started = False
        self.repair_requirements: list[str] = []
        self.reader_done = False
        self.synthesis_done = False
        self.checker_done = False
        self.writer_done = False
        self.retrieval_done = False
        try:
            for _ in range(self.coordinator_steps):
                self.graph.settle()
                ready = self.graph.ready()
                if ready:
                    task = ready[0]
                elif any(t.status == "blocked" for t in self.graph.tasks.values()):
                    state.stop_reason = "task_blocked"
                    break
                else:
                    followup = self._coverage_followup()
                    if followup is not None:
                        task = self._new_task(followup)
                        await self._dispatch(task.id)
                        continue
                    role = self._next_role()
                    if role is None:
                        state.stop_reason = (
                            "requirements_covered"
                            if state.coverage and all(c.status == "covered" for c in state.coverage)
                            else "tasks_completed"
                        )
                        break
                    task = self._new_task(
                        TaskRequest(
                            role=role,
                            instruction="Complete your assigned document research role.",
                            requirement_ids=self.repair_requirements
                            if role in {"reader", "summarizer"}
                            else [],
                        )
                    )
                # Scheduling and dependency handling never call a language model.
                await self._dispatch(task.id)
            else:
                state.stop_reason = "coordinator_step_limit"
            # A final worker may finish on the last dispatch allowed by the limit.
            if self.writer_done and self.checked_revision == self.artifact_revision:
                if all(t.status == "completed" for t in self.graph.tasks.values()):
                    state.stop_reason = (
                        "requirements_covered"
                        if state.coverage and all(c.status == "covered" for c in state.coverage)
                        else "tasks_completed"
                    )
        except ActionRejected as exc:
            state.stop_reason = "task_limit"
            state.warnings.append(str(exc))
        except BudgetExceeded:
            self._close("Budget exhausted before this task completed.")
            raise
        if state.stop_reason not in {"requirements_covered", "tasks_completed"}:
            state.warnings.append("Điều phối đã dừng; báo cáo giữ kết quả kiểm chứng gần nhất.")
        event(
            "orchestrator",
            "completed",
            "Đã kết thúc điều phối bằng code.",
            stop_reason=state.stop_reason,
        )
        self._close("Research concluded before this task completed.")

    def _next_role(self) -> WorkerRole | None:
        """Choose missing/stale prerequisites; worker requests take priority in run()."""
        if not self.retrieval_done:
            return "retrieval"
        if not self.reader_done or self.read_revision != self.source_revision:
            return "reader"
        if not self.synthesis_done or self.draft_revision != self.source_revision:
            return "summarizer"
        if not self.checker_done or self.checked_revision != self.artifact_revision:
            return "fact_checker"
        if not self.writer_done:
            return "writer"
        return None

    def _coverage_followup(self) -> TaskRequest | None:
        """Recover omitted work requests once, using coverage rather than worker completion."""
        if (
            self.coverage_repair_started
            or not self.checker_done
            or self.checked_revision != self.artifact_revision
            or self.searches >= self.request.max_rounds
        ):
            return None
        missing = [c.requirement_id for c in self.state.coverage if c.status != "covered"]
        if not missing:
            return None
        self.coverage_repair_started = True
        self.repair_requirements = missing
        texts = [r.text for r in requirements_for(self.planned) if r.id in missing]
        # Even a repeated candidate set merits one focused reading pass: the first
        # Reader may have overlooked a source unit. Previously verified work stays pinned.
        self.reader_done = self.synthesis_done = self.writer_done = False
        return TaskRequest(
            role="retrieval",
            requirement_ids=missing,
            instruction="Find evidence specifically for these unanswered requirements: "
            + "; ".join(texts)
            + ". Use prior gaps/checks to change the search; do not repeat answered aspects.",
        )

    def _close(self, message: str) -> None:
        for task in self.graph.tasks.values():
            if task.status in {"pending", "running", "waiting"}:
                task.status, task.message = "blocked", message
        for session in self.state.sessions.values():
            if session.status in {"pending", "running", "waiting"}:
                session.status, session.message = "blocked", message

    def _revision(self) -> str:
        return hashlib.sha256(
            json.dumps(
                [
                    self.source_revision,
                    self.artifact_revision,
                    [(t.id, t.status) for t in self.graph.tasks.values()],
                ]
            ).encode()
        ).hexdigest()

    def _progress(self) -> dict[str, object]:
        self.graph.settle()
        usage = current_usage.get()
        return {
            "question": self.request.question,
            "language": self.request.language,
            "requirements": [r.model_dump() for r in requirements_for(self.planned)],
            "coverage": [r.model_dump() for r in self.state.coverage],
            "tasks": [t.model_dump() for t in self.graph.tasks.values()],
            "ready_task_ids": [t.id for t in self.graph.ready()],
            "artifacts": {
                "searches": self.searches,
                "chunks": len(self.active_chunks),
                "reader_done": self.reader_done,
                "synthesis_done": self.synthesis_done,
                "checker_done": self.checker_done,
                "writer_done": self.writer_done,
                "source_revision": self.source_revision,
                "read_revision": self.read_revision,
                "draft_revision": self.draft_revision,
                "artifact_revision": self.artifact_revision,
                "checked_revision": self.checked_revision,
            },
            "retrieval_calls_remaining": self.request.max_rounds - self.searches,
            "tasks_remaining": self.max_tasks - len(self.graph.tasks),
            "model_calls_used": usage.calls if usage else 0,
        }

    def _validate_request(self, request: TaskRequest) -> None:
        if not set(request.requirement_ids).issubset(
            {r.id for r in requirements_for(self.planned)}
        ):
            raise ActionRejected("Unknown requirement ID in task request.")
        if not set(request.claim_ids).issubset({c.id for c in self.proposals + self.state.claims}):
            raise ActionRejected("Unknown claim ID in task request.")
        if not set(request.chunk_ids).issubset(self.state.retrieved):
            raise ActionRejected("Unknown chunk ID in task request.")

    def _new_task(
        self,
        request: TaskRequest,
        role: TaskRequester = "orchestrator",
        parent: str | None = None,
    ) -> RuntimeTask:
        self._validate_request(request)
        task = self.graph.add(request, role, parent)
        self.event(
            role, "completed", "Đã tạo nhiệm vụ cho agent chuyên môn.", task=task.model_dump()
        )
        return task

    async def _dispatch(self, task_id: str) -> None:
        self.graph.settle()
        task = self.graph.tasks.get(task_id)
        if task is None or task not in self.graph.ready():
            raise ActionRejected("Task is not ready; prerequisites must complete first.")
        task.status = "running"
        if task.role == "reader":
            self.inspected.setdefault(task.id, task.chunk_ids or list(self.active_chunks))
        session = self.state.sessions.setdefault(
            task.id, AgentSession(task_id=task.id, role=task.role)
        )

        def child(request: TaskRequest) -> str:
            return self._new_task(request, task.role, task.id).id

        try:
            if task.role == "retrieval":
                await self._worker(task, session, SearchCompletion, self._accept_search, child)
            elif task.role == "reader":
                await self._worker(task, session, ReaderOutput, self._accept_read, child)
            elif task.role == "summarizer":
                await self._worker(task, session, SynthesisOutput, self._accept_synthesis, child)
            elif task.role == "fact_checker":
                await self._worker(task, session, FactCheckOutput, self._accept_check, child)
            else:
                await self._worker(task, session, WriterOutput, self._accept_layout, child)
        except BudgetExceeded:
            raise
        except ProviderError as exc:
            task.status, session.status = "blocked", "blocked"
            task.message, session.message = str(exc), str(exc)
            raise
        task.status, task.message = session.status, session.message
        self.graph.settle()
        status = (
            "completed"
            if session.status == "completed"
            else "running"
            if session.status == "waiting"
            else "failed"
        )
        self.event(task.role, status, "Đã kết thúc hoặc tạm chờ nhiệm vụ.", task=task.model_dump())

    async def _worker(
        self,
        task: RuntimeTask,
        session: AgentSession,
        output: type[T],
        accept: Callable[[T, RuntimeTask], Awaitable[None]],
        child: Callable[[TaskRequest], str],
    ) -> None:
        async def finish(result: T) -> None:
            await accept(result, task)

        def handoff() -> T | None:
            limit = self.request.max_rounds if self.checker_done else 1
            if task.role == "retrieval" and self.searches >= limit:
                return output.model_validate(
                    SearchCompletion(chunk_ids=self.active_chunks).model_dump()
                )
            return None

        try:
            await self._run_worker(task, session, output, finish, child, handoff)
        except ProviderUnavailable:
            # A transient API failure cannot erase already observed local search results.
            # Only source IDs are handed over; Reader and Checker must still run normally.
            if task.role != "retrieval" or not self.active_chunks:
                raise
            result = output.model_validate(
                SearchCompletion(chunk_ids=self.active_chunks).model_dump()
            )
            await finish(result)
            session.status = "completed"
            session.message = "Đã bàn giao nguồn đã tìm được khi API Retrieval tạm gián đoạn."
            self.state.warnings.append(session.message)
            self.event(
                "retrieval",
                "completed",
                session.message,
                task_id=task.id,
                execution="local",
                recovery="observed_sources_only",
                chunk_ids=list(self.active_chunks),
                session=session.model_dump(),
            )

    async def _run_worker(
        self,
        task: RuntimeTask,
        session: AgentSession,
        output: type[T],
        finish: Callable[[T], Awaitable[None]],
        child: Callable[[TaskRequest], str],
        handoff: Callable[[], T | None],
    ) -> None:
        await AgentRuntime(self.llm, self.worker_steps).run(
            session,
            instruction=INSTRUCTIONS[task.role],
            output=output,
            tools=self._worker_tools(task),
            context=lambda: self._context(task),
            revision=self._revision,
            accept=finish,
            request_task=child,
            fallback=lambda: self._worker_fallback(task, output),
            step=self.step,
            event=self.event,
            bound_result=handoff,
        )

    def _chunks(self, ids: list[str] | None = None) -> list[RetrievedChunk]:
        keys = self.active_chunks if ids is None else ids
        if any(cid not in self.state.retrieved for cid in keys):
            raise ActionRejected("Chunk ID is not in this run's observed source scope.")
        return [self.state.retrieved[cid] for cid in keys][:12]

    def _evidence(self) -> list[Evidence]:
        pinned = {eid for c in self.state.claims + self.proposals for eid in c.evidence_ids}
        return list(
            {
                e.id: e
                for e in [
                    *[e for e in self.evidence if e.id in pinned],
                    *self.evidence[:24],
                ]
            }.values()
        )

    def _context(self, task: RuntimeTask) -> dict[str, object]:
        context = self._progress()
        context["assignment"] = task.model_dump()
        context["available_chunks"] = [
            {"id": c.id, "filename": c.filename, "page": c.page, "section": c.section}
            for c in self._chunks()
        ]
        if task.role == "retrieval":
            context["aspects"] = [a.model_dump() for a in self.plan.aspects]
            context["observed_chunks"] = [
                {"id": c.id, "filename": c.filename, "snippet": c.text[:240]}
                for c in self._chunks()
            ]
        elif task.role == "reader":
            context["source_units"] = source_units(self._chunks(self.inspected.get(task.id)))
        elif task.role in {"summarizer", "fact_checker"}:
            context["evidence"] = [e.model_dump(exclude={"regions"}) for e in self._evidence()]
            context["claims"] = [c.model_dump() for c in self.proposals]
            context["candidate_gaps"] = list(enumerate(self.candidate_gaps))
            context["checks"] = [c.model_dump() for c in self.state.checks]
        else:
            supported = {c.claim_id for c in self.state.checks if c.verdict == "supported"}
            context["verified_claims"] = [
                c.model_dump() for c in self.state.claims if c.id in supported
            ]
        return context

    def _worker_fallback(self, task: RuntimeTask, output: type[T]) -> AgentDecision[T]:
        if task.role == "retrieval":
            if self.searches:
                result: BaseModel = SearchCompletion(chunk_ids=self.active_chunks)
            else:
                return AgentDecision(
                    action="use_tool",
                    tool="search",
                    arguments=ToolArguments(
                        queries=[a.query for a in self.plan.aspects[:4]],
                        search_mode=self.request.retrieval_mode,
                    ),
                    reason="Search the fixed corpus in the offline demo.",
                )
        elif task.role == "reader":
            result = self.reader._extract(
                self.request.question, self._chunks(task.chunk_ids or None)
            )
        elif task.role == "summarizer":
            result = SynthesisOutput(
                claims=[
                    RawClaim(text=e.quote, evidence_ids=[e.id], requirement_ids=e.requirement_ids)
                    for e in self._evidence()[:12]
                    if 10 <= len(e.quote) <= 1000
                ]
            )
        elif task.role == "fact_checker":
            result = self.checker._extractive_checks(self.proposals, self._evidence())
        else:
            supported = {c.claim_id for c in self.state.checks if c.verdict == "supported"}
            result = WriterOutput(
                title="Research notes",
                sections=self.writer._sections(
                    [c for c in self.state.claims if c.id in supported],
                    self.planned,
                ),
            )
        return AgentDecision(
            action="finish",
            result=output.model_validate(result.model_dump()),
            reason="Submit the structured offline demonstration result.",
        )

    def _remember_chunks(self, chunks: list[RetrievedChunk]) -> None:
        if any(c.document_id not in self.request.document_ids for c in chunks):
            raise ActionRejected("Tool returned a source outside the frozen document scope.")
        # Section reads can return the indexed child of a context already expanded for Reader.
        # Keep the exact larger window so older evidence locators remain usable in the UI.
        preserved = []
        for c in chunks:
            old = self.state.retrieved.get(c.id)
            if (
                old is not None
                and old.start <= c.start
                and old.end >= c.end
                and (old.text[c.start - old.start : c.end - old.start] == c.text)
            ):
                preserved.append(old)
            else:
                preserved.append(c)
        chunks = preserved
        changed = any(
            c.id not in self.state.retrieved or self.state.retrieved[c.id].text != c.text
            for c in chunks
        )
        self.state.retrieved.update({c.id: c for c in chunks})
        self.active_chunks = merge_source_candidates(self.active_chunks, [c.id for c in chunks])
        if changed:
            self.source_revision += 1
            self.artifact_revision += 1
            self.state.report_layout = None
            self.writer_done = False

    def _worker_tools(self, task: RuntimeTask) -> list[AgentTool]:
        async def search(args: ToolArguments) -> dict[str, object]:
            if not args.queries:
                raise ActionRejected("Search requires a nonempty query list.")
            if self.searches >= self.request.max_rounds:
                raise ActionRejected("The global retrieval limit has been reached.")
            self.searches += 1
            self.state.rounds = self.searches
            plan, chunks, notes = await self.retrieval.run(
                self.request,
                [],
                self.plan,
                set(task.requirement_ids) or None,
                set(self.state.retrieved),
                queries_override=args.queries,
                mode_override=args.search_mode or self.request.retrieval_mode,
            )
            self.state.warnings.extend(notes)
            self._remember_chunks(chunks)
            return {
                "queries": plan.queries,
                "chunks": [
                    {"id": c.id, "filename": c.filename, "snippet": c.text[:320]} for c in chunks
                ],
                "warnings": notes,
                "search_diagnostics": getattr(self.retrieval.search, "diagnostics", lambda: {})(),
            }

        async def read(args: ToolArguments) -> dict[str, object]:
            chunks = self._chunks(args.chunk_ids or None)
            if task.role == "reader":
                self.inspected[task.id] = list(
                    dict.fromkeys([c.id for c in chunks] + self.inspected.get(task.id, []))
                )[:12]
            return {"source_units": source_units(self._chunks([c.id for c in chunks]))}

        async def expand(args: ToolArguments) -> dict[str, object]:
            if not args.chunk_ids:
                raise ActionRejected("Choose real observed chunk IDs to open context.")
            chunks = self._chunks(args.chunk_ids)
            method = getattr(self.retrieval.search, "expand_context", None)
            if method:
                chunks = await method(chunks, self.request.document_ids)
            self._remember_chunks(chunks)
            self.inspected[task.id] = list(
                dict.fromkeys([c.id for c in chunks] + self.inspected.get(task.id, []))
            )[:12]
            return {"source_units": source_units(self._chunks([c.id for c in chunks]))}

        async def section(args: ToolArguments) -> dict[str, object]:
            if not args.chunk_ids:
                raise ActionRejected("A section must be anchored to observed chunk IDs.")
            seeds = self._chunks(args.chunk_ids)
            method = getattr(self.retrieval.search, "read_section", None)
            if method is None:
                raise ActionRejected("This search adapter does not support section reading.")
            chunks = await method(seeds, self.request.document_ids)
            self._remember_chunks(chunks)
            self.inspected[task.id] = list(
                dict.fromkeys([c.id for c in chunks] + self.inspected.get(task.id, []))
            )[:12]
            return {"source_units": source_units(self._chunks([c.id for c in chunks]))}

        async def evidence(args: ToolArguments) -> dict[str, object]:
            lookup = {e.id: e for e in self._evidence()}
            if not set(args.evidence_ids).issubset(lookup):
                raise ActionRejected("Unknown evidence ID.")
            return {
                "evidence": [
                    lookup[eid].model_dump(exclude={"regions"})
                    for eid in (args.evidence_ids or list(lookup))
                ]
            }

        async def claims(args: ToolArguments) -> dict[str, object]:
            items = self.state.claims if task.role == "writer" else self.proposals
            if task.role == "writer":
                valid = {c.claim_id for c in self.state.checks if c.verdict == "supported"}
                items = [c for c in items if c.id in valid]
            lookup = {c.id: c for c in items}
            if not set(args.claim_ids).issubset(lookup):
                raise ActionRejected("Claim ID is not available to this role.")
            return {
                "claims": [lookup[cid].model_dump() for cid in (args.claim_ids or list(lookup))]
            }

        async def numeric(args: ToolArguments) -> dict[str, object]:
            lookup = {c.id: c for c in self.proposals}
            evidence_lookup = {e.id: e for e in self._evidence()}
            if not set(args.claim_ids).issubset(lookup):
                raise ActionRejected("Unknown claim ID for number inspection.")
            rows = []
            for cid in args.claim_ids or list(lookup):
                c = lookup[cid]
                quotes = [
                    evidence_lookup[eid].quote for eid in c.evidence_ids if eid in evidence_lookup
                ]
                quoted = " ".join(quotes)
                reason = numeric_guard_reason(c.text, quotes)
                rows.append(
                    {
                        "claim_id": cid,
                        "claim_numbers": sorted(numbers(c.text)),
                        "source_numbers": sorted(numbers(quoted)),
                        "literal_numbers_present": numbers(c.text).issubset(numbers(quoted)),
                        "numeric_guard_passed": not reason,
                        "guard_reason": reason,
                    }
                )
            return {
                "number_checks": rows,
                "note": "Explicit dates are normalized; other numbers stay literal. "
                "This is not a semantic verdict; inspect entities, units and conditions.",
            }

        async def progress(_: ToolArguments) -> dict[str, object]:
            return self._progress()

        tools = {
            "search": AgentTool(
                "search",
                "Search the scoped corpus with selected queries and lexical/hybrid mode.",
                frozenset({"queries", "search_mode"}),
                search,
            ),
            "inspect_sources": AgentTool(
                "inspect_sources",
                "Inspect units of already observed chunks.",
                frozenset({"chunk_ids"}),
                read,
            ),
            "read_units": AgentTool(
                "read_units",
                "Inspect exact units from observed source chunks.",
                frozenset({"chunk_ids"}),
                read,
            ),
            "read_context": AgentTool(
                "read_context",
                "Open adjacent context around real chunk IDs.",
                frozenset({"chunk_ids"}),
                expand,
            ),
            "read_section": AgentTool(
                "read_section",
                "Read a bounded portion of the same source section.",
                frozenset({"chunk_ids"}),
                section,
            ),
            "inspect_evidence": AgentTool(
                "inspect_evidence",
                "Inspect exact evidence quotes and provenance.",
                frozenset({"evidence_ids"}),
                evidence,
            ),
            "inspect_claims": AgentTool(
                "inspect_claims",
                "Inspect available claim texts and citations.",
                frozenset({"claim_ids"}),
                claims,
            ),
            "check_numbers": AgentTool(
                "check_numbers",
                "Check equivalent explicit dates and literal numbers; not a semantic verdict.",
                frozenset({"claim_ids"}),
                numeric,
            ),
            "inspect_progress": AgentTool(
                "inspect_progress",
                "Inspect artifact versions, coverage and task dependencies.",
                frozenset(),
                progress,
            ),
        }
        allowed = {
            "retrieval": ["search", "inspect_sources"],
            "reader": ["read_units", "read_context", "read_section"],
            "summarizer": ["inspect_evidence", "inspect_claims"],
            "fact_checker": ["inspect_evidence", "inspect_claims", "check_numbers"],
            "writer": ["inspect_claims", "inspect_progress"],
        }
        return [tools[name] for name in allowed[task.role]]

    async def _accept_search(self, result: SearchCompletion, _: RuntimeTask) -> None:
        if not set(result.chunk_ids).issubset(self.state.retrieved):
            raise ActionRejected("Retrieval may finish with observed chunk IDs only.")
        self.active_chunks = list(dict.fromkeys(result.chunk_ids))
        self.retrieval_done = True

    async def _accept_read(self, result: ReaderOutput, task: RuntimeTask) -> None:
        chunks = self._chunks(self.inspected.get(task.id))
        evidence, gaps, notes = self.reader.validate_output(
            self.request.question,
            chunks,
            result,
            self.planned.requirements,
        )
        if result.evidence and not evidence:
            raise ActionRejected("No selected evidence has a valid exact source unit or quote.")
        ledger = {(e.document_id, e.page, e.quote): e for e in self.evidence}
        for e in evidence:
            key = (e.document_id, e.page, e.quote)
            previous = ledger.get(key)
            ledger[key] = (
                previous.model_copy(
                    update={
                        "requirement_ids": list(
                            dict.fromkeys(previous.requirement_ids + e.requirement_ids)
                        ),
                    }
                )
                if previous
                else e
            )
        fresh = {(e.document_id, e.page, e.quote) for e in evidence}
        candidate_evidence = [e for key, e in ledger.items() if key in fresh]
        candidate_evidence += [e for key, e in ledger.items() if key not in fresh]
        if len(candidate_evidence) > 96:
            raise ActionRejected("Evidence ledger is full; narrow the task instead of adding more.")
        self.evidence = candidate_evidence
        self.candidate_gaps = list(dict.fromkeys(gaps))[:16]
        self.state.warnings.extend(notes)
        self.reader_done, self.read_revision = True, self.source_revision
        self.synthesis_done = False
        self.artifact_revision += 1
        self.state.report_layout = None
        self.writer_done = False

    async def _accept_synthesis(self, result: SynthesisOutput, task: RuntimeTask) -> None:
        if self.read_revision != self.source_revision:
            raise ActionRejected("Reader must inspect the current source context before synthesis.")
        requirements = requirements_for(self.planned)
        focused = [
            r for r in requirements if not task.requirement_ids or r.id in task.requirement_ids
        ]
        claims, gaps, notes = self.summarizer.validate_output(result, self._evidence(), focused)
        if result.claims and not claims:
            raise ActionRejected("Every proposed claim references invalid evidence IDs.")
        supported = {c.claim_id for c in self.state.checks if c.verdict == "supported"}
        retained = [
            c for c in self.state.claims if c.id in supported and c.id not in task.claim_ids
        ]
        protected = [
            c
            for r in requirements
            if task.requirement_ids and r.id not in task.requirement_ids
            for c in [c for c in retained if r.id in c.requirement_ids][:1]
        ]
        merged: dict[str, Claim] = {}
        for c in [*protected, *claims, *retained]:
            old = merged.get(c.id)
            merged[c.id] = (
                c.model_copy(
                    update={
                        "requirement_ids": list(
                            dict.fromkeys(old.requirement_ids + c.requirement_ids)
                        ),
                    }
                )
                if old
                else c
            )
        self.proposals = list(merged.values())[:16]
        self.candidate_gaps = list(dict.fromkeys([*self.candidate_gaps, *gaps]))[:16]
        self.state.warnings.extend(notes)
        self.synthesis_done, self.draft_revision = True, self.source_revision
        self.artifact_revision += 1
        self.state.report_layout = None
        self.writer_done = False

    async def _accept_check(self, result: FactCheckOutput, task: RuntimeTask) -> None:
        if not self.synthesis_done or self.draft_revision != self.source_revision:
            raise ActionRejected("Current sources need Reader/Summarizer work before verification.")
        requirements = requirements_for(self.planned)
        evidence = self._evidence()
        validated = self.checker.validate_output(result, self.proposals, evidence, requirements)
        coverage, gaps = reconcile_gaps(
            requirements,
            validated.coverage,
            self.candidate_gaps,
            validated.gap_checks,
            evidence,
        )
        # This is the only place that advances the reportable factual snapshot.
        self.state.evidence, self.state.claims = list(evidence), list(self.proposals)
        self.state.checks, self.state.coverage = validated.checks, coverage
        self.state.gap_details = gaps
        self.state.gaps = gap_messages(gaps, requirements, self.request.language)
        self.checked_revision, self.checker_done = self.artifact_revision, True
        self.state.report_layout, self.writer_done = None, False
        await self.state.save_checkpoint()
        # Preserve this verified snapshot even if the repair hits the global budget.
        # One feedback opportunity per artifact avoids silently completing after a
        # structural downgrade, without repeatedly spending calls on the same draft.
        proposed_supported = {c.claim_id for c in result.checks if c.verdict == "supported"}
        rejected = [
            c
            for c in validated.checks
            if c.claim_id in proposed_supported and c.verdict != "supported"
        ]
        session = self.state.sessions[task.id]
        feedback = (
            rejected
            and not validated.requests
            and self.guard_feedback_revision != self.artifact_revision
            and session.steps < self.worker_steps
        )
        for item in validated.requests:
            roles: dict[str, WorkerRole] = {
                "retrieve": "retrieval",
                "read_context": "reader",
                "revise": "summarizer",
                "verify": "fact_checker",
            }
            role = roles[item.action]
            followup = TaskRequest(
                role=role,
                instruction=item.need,
                requirement_ids=item.requirement_ids,
                claim_ids=item.claim_ids,
                chunk_ids=item.chunk_ids,
                depends_on=[task.id],
            )
            if any(
                t.role == role and t.instruction == item.need and t.status in {"pending", "waiting"}
                for t in self.graph.tasks.values()
            ):
                continue
            try:
                self._new_task(followup, "fact_checker")
            except ActionRejected as exc:
                self.state.warnings.append(str(exc))
        self.event(
            "fact_checker",
            "completed",
            "Đã cập nhật snapshot kiểm chứng.",
            checks=[c.model_dump() for c in validated.checks],
            coverage=[c.model_dump() for c in coverage],
            revision=self.checked_revision,
        )
        if feedback:
            self.guard_feedback_revision = self.artifact_revision
            reasons = "; ".join(f"{c.claim_id}: {c.explanation}" for c in rejected)
            raise ActionRejected(
                "Structural guards rejected supported verdicts: "
                + reasons
                + ". Inspect current checks, then correct verdicts/coverage or request a "
                "specialist to repair the claim. Do not repeat the unchanged supported verdict."
            )

    async def _accept_layout(self, result: WriterOutput, _: RuntimeTask) -> None:
        if not self.checker_done or self.checked_revision != self.artifact_revision:
            raise ActionRejected(
                "Writer cannot publish a layout against an unchecked artifact version."
            )
        supported = {c.claim_id for c in self.state.checks if c.verdict == "supported"}
        ids = [block.claim_id for section in result.sections for block in section.blocks]
        if not set(ids).issubset(supported) or len(ids) != len(set(ids)):
            raise ActionRejected(
                "Writer may order each verified claim once; unknown or unchecked ID."
            )
        self.state.report_layout, self.writer_done = result, True
