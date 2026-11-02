"""Two explicit requirements, independent source workers and a terminal join.

The existing five specialist agents perform the work. Only the source branches
fan out; synthesis/checking/writing retain their factual dependencies.
"""

import asyncio
from collections.abc import Callable
from typing import Any, Literal

from research_assist.agents.fact_checker import FactCheckerAgent
from research_assist.agents.reader import ReaderAgent
from research_assist.agents.retrieval import RetrievalAgent
from research_assist.agents.summarizer import SummarizerAgent
from research_assist.agents.writer import WriterAgent
from research_assist.application.agent_runtime import Event, Step
from research_assist.application.citation_audit import (
    audit_citations,
    remove_rejected_claims,
    verification_snapshot,
)
from research_assist.application.research_state import ResearchState
from research_assist.application.task_execution import (
    TaskExecutionContext,
    TaskExecutor,
    fingerprint,
)
from research_assist.application.task_graph import TaskGraph
from research_assist.domain.coverage import (
    gap_messages,
    reconcile_gaps,
    report_coverage,
    requirements_for,
)
from research_assist.domain.errors import (
    ActionRejected,
    BudgetExceeded,
    ProviderError,
    TaskBudgetExceeded,
)
from research_assist.domain.metadata import EVIDENCE_POLICY
from research_assist.domain.models import (
    AgentSession,
    ClaimEvidenceLink,
    Evidence,
    EvidenceUsage,
    QuestionPlan,
    ResearchRequest,
    ResearchTaskState,
    RetrievedChunk,
    RuntimeTask,
    TaskBudget,
    TaskRequest,
    TaskScope,
)
from research_assist.infrastructure.llm import current_task_budget, current_usage


class ScopedSources:
    """A capability for one execution; no task may read the run-wide chunk pool."""

    def __init__(
        self, search: Any, scope: TaskScope, budget: TaskBudget, reserve: Callable[[], None]
    ) -> None:
        self.search_tool, self.scope, self.budget = search, scope, budget
        self.reserve = reserve
        self.observed: dict[str, RetrievedChunk] = {}
        self.expansions: list[TaskScope] = []

    def validate(self, chunk: RetrievedChunk) -> None:
        if chunk.document_id not in self.scope.allowed_document_ids:
            raise ActionRejected("Search returned a document outside the task scope.")
        # Frozen adapters expose indexed source windows. Verify the entire returned
        # span against their union, including any context added around a seed.
        frozen = getattr(self.search_tool, "frozen_chunks", None)
        if frozen is not None:
            anchor = frozen.get(chunk.id)
            if anchor is None or (anchor.document_id, anchor.page) != (
                chunk.document_id,
                chunk.page,
            ):
                raise ActionRejected("Unknown source anchor in the frozen snapshot.")
            covered: set[int] = set()
            for source in frozen.values():
                if (source.document_id, source.page) != (chunk.document_id, chunk.page):
                    continue
                left, right = max(source.start, chunk.start), min(source.end, chunk.end)
                if left < right:
                    if (
                        source.text[left - source.start : right - source.start]
                        != chunk.text[left - chunk.start : right - chunk.start]
                    ):
                        raise ActionRejected("Source content differs from the pinned snapshot.")
                    covered.update(range(left, right))
            if len(covered) != len(chunk.text) or chunk.end - chunk.start != len(chunk.text):
                raise ActionRejected("Expanded source contains an unobserved span.")

    async def search(
        self, queries: list[str], document_ids: list[str], top_k: int, mode: str
    ) -> tuple[list[RetrievedChunk], list[str]]:
        if set(document_ids) != set(self.scope.allowed_document_ids):
            raise ActionRejected("Search attempted to change document scope.")
        self.reserve()
        chunks, warnings = await self.search_tool.search(queries, document_ids, top_k, mode)
        for chunk in chunks:
            self.validate(chunk)
            self.observed[chunk.id] = chunk.model_copy(deep=True)
        return chunks, warnings

    async def expand_context(
        self, chunks: list[RetrievedChunk], document_ids: list[str]
    ) -> list[RetrievedChunk]:
        if set(document_ids) != set(self.scope.allowed_document_ids):
            raise ActionRejected("Context attempted to change document scope.")
        if any(c.id not in self.observed for c in chunks):
            raise ActionRejected("Context expansion must be anchored to task-observed sources.")
        before = self.scope.model_copy(
            update={
                "allowed_chunk_ids": tuple(c.id for c in chunks),
                "scope_version": fingerprint([c.model_dump() for c in chunks]),
            }
        )
        expand = getattr(self.search_tool, "expand_context", None)
        result = await expand(chunks, document_ids) if expand else chunks
        seeds = {c.id: c for c in chunks}
        for chunk in result:
            seed = seeds.get(chunk.id)
            if seed is None or (seed.document_id, seed.page) != (chunk.document_id, chunk.page):
                raise ActionRejected("Context expansion changed its source anchor.")
            if not (chunk.start <= seed.start and chunk.end >= seed.end):
                raise ActionRejected("Context expansion lost its source anchor.")
            if chunk.text[seed.start - chunk.start : seed.end - chunk.start] != seed.text:
                raise ActionRejected("Context expansion changed source text.")
            self.validate(chunk)
            self.observed[chunk.id] = chunk.model_copy(deep=True)
        after = self.scope.model_copy(
            update={
                "allowed_chunk_ids": tuple(c.id for c in result),
                "scope_version": fingerprint([c.model_dump() for c in result]),
            }
        )
        self.expansions.extend([before, after])
        return result


class ParallelResearchCoordinator:
    def __init__(
        self,
        retrieval: RetrievalAgent,
        reader: ReaderAgent,
        summarizer: SummarizerAgent,
        checker: FactCheckerAgent,
        writer: WriterAgent,
        parallelism: int = 2,
        max_tasks: int = 16,
        max_steps: int = 16,
        retries: int = 1,
        task_calls: int = 6,
        task_tokens: int = 20000,
    ) -> None:
        self.retrieval, self.reader, self.summarizer = retrieval, reader, summarizer
        self.checker, self.writer = checker, writer
        self.parallelism, self.max_tasks, self.max_steps = parallelism, max_tasks, max_steps
        self.retries, self.task_calls, self.task_tokens = retries, task_calls, task_tokens
        self.executor = TaskExecutor()
        self.chunks: dict[str, dict[str, RetrievedChunk]] = {}
        self.sources: dict[str, list[RetrievedChunk]] = {}
        self.versions: dict[str, str] = {}
        self.search_calls = 0
        self.approved_snapshot = ""

    async def run(
        self,
        request: ResearchRequest,
        plan: QuestionPlan,
        state: ResearchState,
        step: Step,
        event: Event,
    ) -> None:
        if len(request.requirements) != 2:
            raise ActionRejected("Parallel MVP requires two explicit requirements.")
        self.request, self.plan, self.state, self.step, self.event = (
            request,
            plan,
            state,
            step,
            event,
        )
        self.graph = TaskGraph(state.task_graph, self.max_tasks)
        usage = current_usage.get()
        if usage:
            usage.strict_reservations = True
        freeze = getattr(self.retrieval.search, "freeze", None)
        self.search = await freeze(request.document_ids) if freeze else self.retrieval.search
        self.versions = dict(getattr(self.search, "source_versions", {}))
        for doc in request.document_ids:
            self.versions.setdefault(doc, fingerprint([doc, "adapter-snapshot"]))
        readers = []
        for requirement in requirements_for(request):
            retrieval = self.add("retrieval", [requirement.id])
            reader = self.add("reader", [requirement.id], [retrieval.id], "ON_SUCCESS")
            readers.append(reader.id)
        summary = self.add("summarizer", ["rq_1", "rq_2"], readers, "ON_TERMINAL")
        checker = self.add("fact_checker", ["rq_1", "rq_2"], [summary.id])
        writer = self.add("writer", ["rq_1", "rq_2"], [checker.id])
        # Audit is deterministic, represented as a writer-role service task.
        audit = self.add("writer", ["rq_1", "rq_2"], [writer.id])
        self.audit_task_id = audit.id
        running: dict[asyncio.Task[None], RuntimeTask] = {}
        dispatched = 0
        try:
            while self.executor.active:
                self.graph.settle()
                for task in self.graph.ready():
                    if len(running) >= self.parallelism or dispatched >= self.max_steps:
                        break
                    task.status = "running"
                    running[asyncio.create_task(self.dispatch(task), name=task.id)] = task
                    dispatched += 1
                if not running:
                    break
                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for future in done:
                    running.pop(future)
                    if future.cancelled():
                        # A task-local cancellation is terminal for its own branch.
                        # Cancellation of this scheduler is handled by the finally block.
                        continue
                    future.result()  # Global budget and programming failures stop the run.
            state.stop_reason = (
                (
                    "requirements_covered"
                    if all(c.status == "covered" for c in state.coverage)
                    else "tasks_completed"
                )
                if audit.status == "completed"
                else "task_limit"
            )
        finally:
            self.executor.stop(list(state.task_graph.values()))
            for future in running:
                future.cancel()
            await asyncio.gather(*running, return_exceptions=True)

    def add(
        self,
        role: Any,
        requirements: list[str],
        dependencies: list[str] | None = None,
        condition: Literal["ON_SUCCESS", "ON_TERMINAL"] = "ON_SUCCESS",
    ) -> RuntimeTask:
        dependencies = dependencies or []
        task = self.graph.add(
            TaskRequest(
                role=role,
                instruction="Complete this scoped research task.",
                requirement_ids=requirements,
                depends_on=dependencies,
                dependency_conditions={d: condition for d in dependencies},
            ),
            "orchestrator",
        )
        self.state.research_tasks[task.id] = ResearchTaskState(
            task_id=task.id,
            scope=TaskScope(
                task_id=task.id,
                requirement_ids=tuple(requirements),
                allowed_document_ids=tuple(self.request.document_ids),
            ),
            budget=TaskBudget(
                max_search_calls=self.request.max_rounds,
                max_model_calls=self.task_calls,
                max_tokens=self.task_tokens,
            ),
        )
        return task

    def task_chunks(self, task_id: str, ids: list[str] | None = None) -> list[RetrievedChunk]:
        state = self.state.research_tasks[task_id]
        keys = list(state.scope.allowed_chunk_ids) if ids is None else ids
        if not set(keys).issubset(state.scope.allowed_chunk_ids):
            raise ActionRejected("Reader requested a chunk outside its task scope.")
        return [self.chunks[task_id][key].model_copy(deep=True) for key in keys]

    def dependency(self, task: RuntimeTask) -> str:
        state = self.state.research_tasks[task.id]
        model = {
            "retrieval": self.retrieval.llm,
            "reader": self.reader.llm,
            "summarizer": self.summarizer.llm,
            "fact_checker": self.checker.llm,
            "writer": self.writer.llm,
        }[task.role]
        common = [
            state.scope.model_dump(),
            self.request.question,
            EVIDENCE_POLICY,
            model.model,
            model.name,
        ]
        if task.role in {"retrieval", "reader"}:
            docs = (
                state.scope.allowed_document_ids
                if task.role == "retrieval"
                else {c.document_id for c in self.chunks.get(task.id, {}).values()}
            )
            return fingerprint(
                [
                    common,
                    {d: self.versions[d] for d in docs},
                    [c.model_dump() for c in self.chunks.get(task.id, {}).values()],
                    [
                        r.model_dump()
                        for r in requirements_for(self.request)
                        if r.id in task.requirement_ids
                    ],
                    self.request.retrieval_mode,
                    getattr(getattr(self.search, "embedder", None), "fingerprint", None),
                    self.retrieval.candidate_limit,
                    getattr(self.retrieval.reranker, "model_name", None),
                ]
            )
        return fingerprint(
            [
                common,
                verification_snapshot(self.state),
                [t.model_dump() for t in self.state.task_graph.values() if t.id in task.depends_on],
            ]
        )

    def audit_now(self) -> None:
        """Audit before checkpoints as well as before final publication."""
        ids = [c.claim_id for c in self.state.checks if c.verdict == "supported"]
        audit = audit_citations(
            self.state, self.sources, self.versions, self.approved_snapshot, ids
        )
        previous = self.state.citation_audit
        audit.rejected_claim_ids = list(
            dict.fromkeys(previous.rejected_claim_ids + audit.rejected_claim_ids)
        )
        audit.reasons = list(dict.fromkeys(previous.reasons + audit.reasons))
        self.state.citation_audit = audit
        remove_rejected_claims(self.state, audit)
        self.state.coverage = report_coverage(self.state.coverage, audit.accepted_claim_ids)
        self.approved_snapshot = verification_snapshot(self.state)

    async def dispatch(self, task: RuntimeTask) -> None:
        local = self.state.research_tasks[task.id]
        session = AgentSession(task_id=task.id, role=task.role, status="running")
        self.state.sessions[task.id] = session
        token = current_task_budget.set(local.budget)
        try:
            for attempt in range(self.retries + 1):
                if not self.executor.active:
                    return
                task.status = "running"
                task.attempts += 1
                session.steps += 1
                self.prepare_scope(task)
                try:
                    await self.executor.execute(
                        task,
                        local,
                        lambda: self.dependency(task),
                        lambda ctx: self.work(task, ctx),
                        lambda result: self.publish(task, result),
                    )
                    session.status = "completed"
                    if task.role == "fact_checker" or task.id == self.audit_task_id:
                        self.audit_now()
                        await self.state.save_checkpoint()
                    break
                except BudgetExceeded:
                    raise
                except (ProviderError, ActionRejected) as exc:
                    if not self.executor.is_running(task):
                        session.status = "cancelled"
                        return
                    task.message = str(exc)
                    local.warnings.append(str(exc))
                    if attempt == self.retries or isinstance(exc, TaskBudgetExceeded):
                        task.status, session.status = "failed", "failed"
                        self.state.warnings.append(f"{task.id}: {exc}")
                        break
                    task.status = "retry_waiting"
                    task.generation += 1
                    await asyncio.sleep(0)  # Yield; HTTP retries already implement backoff.
        except asyncio.CancelledError:
            task.generation += 1
            task.status, session.status = "cancelled", "cancelled"
            raise
        finally:
            current_task_budget.reset(token)
            self.event(
                task.role,
                "completed" if task.status == "completed" else "failed",
                "Scoped research task ended.",
                task=task.model_dump(),
                research_task=local.model_dump(),
            )

    def prepare_scope(self, task: RuntimeTask) -> None:
        if task.role != "reader":
            return
        parent = self.state.research_tasks[task.depends_on[0]]
        local = self.state.research_tasks[task.id]
        local.scope = local.scope.model_copy(
            update={
                "allowed_chunk_ids": parent.scope.allowed_chunk_ids,
                "scope_version": parent.scope.scope_version,
            }
        )
        self.chunks[task.id] = {c.id: c for c in self.task_chunks(parent.task_id)}

    async def work(self, task: RuntimeTask, context: TaskExecutionContext) -> Any:
        local = self.state.research_tasks[task.id]
        if task.role == "retrieval":

            def reserve() -> None:
                if local.budget.search_calls >= local.budget.max_search_calls:
                    raise TaskBudgetExceeded("Task search budget exhausted.")
                if self.search_calls >= 2 * self.request.max_rounds:
                    raise BudgetExceeded("Run search budget exhausted.")
                local.budget.search_calls += 1
                self.search_calls += 1

            scoped = ScopedSources(self.search, local.scope, local.budget, reserve)
            agent = RetrievalAgent(
                self.retrieval.llm,
                scoped,
                self.retrieval.reranker,
                self.retrieval.candidate_limit,
                self.retrieval.reranker_fallback,
            )
            index = int(task.requirement_ids[0].removeprefix("rq_")) - 1
            request = self.request.model_copy(
                update={
                    "question": self.request.requirements[index] + "\n" + self.request.question,
                    "requirements": [self.request.requirements[index]],
                }
            )
            result = await self.step(
                "retrieval",
                "Search a scoped requirement.",
                agent.run(
                    request,
                    [],
                    QuestionPlan(aspects=[self.plan.aspects[index]]),
                    queries_override=[request.question],
                ),
            )
            return result, scoped.expansions
        if task.role == "reader":
            chunks = self.task_chunks(task.id, list(context.allowed_chunk_ids))
            texts = [
                self.request.requirements[int(r.removeprefix("rq_")) - 1]
                for r in task.requirement_ids
            ]
            return await self.step(
                "reader",
                "Read task-owned source windows.",
                self.reader.run(
                    texts[0] + "\n" + self.request.question,
                    chunks,
                    texts,
                ),
            )
        if task.role == "summarizer":
            return await self.step(
                "summarizer",
                "Merge terminal research branches.",
                self.summarizer.run(
                    self.request.question,
                    self.evidence_view(),
                    self.request.language,
                    requirements_for(self.request),
                ),
            )
        if task.role == "fact_checker":
            return await self.step(
                "fact_checker",
                "Verify merged evidence and coverage.",
                self.checker.assess(
                    self.request.question,
                    [c.model_copy(deep=True) for c in self.state.claims],
                    self.evidence_view(),
                    requirements_for(self.request),
                    self.state.gaps,
                ),
            )
        report = await self.writer.compose(
            self.request,
            self.state.claims,
            self.state.checks,
            self.state.evidence,
            self.state.gaps,
            self.state.gap_details,
            deterministic=True,
        )
        if task.id == self.audit_task_id:
            return audit_citations(
                self.state, self.sources, self.versions, self.approved_snapshot, report.claim_ids
            )
        return report

    def evidence_view(self) -> list[Evidence]:
        """Attach usage metadata to prompt copies, never mutate ledger evidence."""
        return [
            e.model_copy(
                deep=True,
                update={
                    "requirement_ids": list(
                        dict.fromkeys(
                            u.requirement_id
                            for u in self.state.evidence_usages
                            if u.evidence_id == e.id
                        )
                    ),
                },
            )
            for e in self.state.evidence
        ]

    def publish(self, task: RuntimeTask, result: Any) -> None:
        local = self.state.research_tasks[task.id]
        if task.role == "retrieval":
            (plan, chunks, warnings), expansions = result
            if len(chunks) > self.request.top_k:
                raise ActionRejected("Task chunk limit exceeded.")
            self.chunks[task.id] = {c.id: c.model_copy(deep=True) for c in chunks}
            local.chunk_ids = list(self.chunks[task.id])
            local.scope = local.scope.model_copy(
                update={
                    "allowed_chunk_ids": tuple(local.chunk_ids),
                    "scope_version": fingerprint([c.model_dump() for c in chunks]),
                }
            )
            local.scope_history.extend(expansions or [local.scope])
            local.warnings.extend(warnings)
            for chunk in chunks:
                self.state.retrieved.setdefault(chunk.id, chunk.model_copy(deep=True))
                self.sources.setdefault(chunk.id, []).append(chunk.model_copy(deep=True))
                self.versions.setdefault(
                    chunk.document_id,
                    fingerprint(
                        [
                            chunk.document_id,
                            chunk.text,
                            chunk.start,
                            chunk.end,
                        ]
                    ),
                )
            self.state.history.append({"task_id": task.id, "queries": plan.queries})
            return
        if task.role == "reader":
            evidence, gaps, warnings = result
            if len(evidence) > 24:
                raise ActionRejected("Task evidence limit exceeded.")
            ledger = {e.id: e for e in self.state.evidence}
            usages = list(self.state.evidence_usages)
            evidence_ids = []
            for incoming in evidence:
                if incoming.chunk_id not in local.scope.allowed_chunk_ids:
                    raise ActionRejected("Reader submitted evidence outside its task scope.")
                version = self.versions[incoming.document_id]
                identity = fingerprint(
                    [
                        incoming.document_id,
                        version,
                        incoming.page,
                        incoming.start,
                        incoming.end,
                        incoming.quote,
                    ]
                )
                ev = incoming.model_copy(
                    update={
                        "id": "ev_" + identity[:24],
                        "document_version": version,
                        "requirement_ids": [],
                    }
                )
                ledger.setdefault(ev.id, ev)
                evidence_ids.append(ev.id)
                for rid in task.requirement_ids:
                    usage = EvidenceUsage(evidence_id=ev.id, task_id=task.id, requirement_id=rid)
                    if usage not in usages:
                        usages.append(usage)
            self.state.evidence = list(ledger.values())
            self.state.evidence_usages = usages
            local.evidence_ids = list(dict.fromkeys(evidence_ids))
            local.inspected_chunk_ids = list(local.scope.allowed_chunk_ids)
            local.warnings.extend(warnings)
            self.state.warnings.extend(warnings)
            # Free-text absence claims are never published as established facts.
            self.state.history.append({"task_id": task.id, "candidate_gaps": gaps})
            return
        if task.role == "summarizer":
            claims, gaps, warnings = result
            self.state.claims, self.state.gaps = claims, gaps
            self.state.warnings.extend(warnings)
            self.state.claim_evidence_links = [
                ClaimEvidenceLink(claim_id=c.id, evidence_id=eid)
                for c in claims
                for eid in c.evidence_ids
            ]
            return
        if task.role == "fact_checker":
            self.state.checks = result.checks
            self.state.coverage, self.state.gap_details = reconcile_gaps(
                requirements_for(self.request),
                result.coverage,
                self.state.gaps,
                result.gap_checks,
                self.state.evidence,
            )
            self.state.gaps = gap_messages(
                self.state.gap_details, requirements_for(self.request), self.request.language
            )
            self.approved_snapshot = verification_snapshot(self.state)
            return
        if task.id == self.audit_task_id:
            result.rejected_claim_ids = list(
                dict.fromkeys(
                    self.state.citation_audit.rejected_claim_ids + result.rejected_claim_ids
                )
            )
            result.reasons = list(dict.fromkeys(self.state.citation_audit.reasons + result.reasons))
            self.state.citation_audit = result
            remove_rejected_claims(self.state, result)
            self.state.coverage = report_coverage(self.state.coverage, result.accepted_claim_ids)
            self.event(
                "citation_audit",
                "completed",
                "Citation audit completed.",
                audit=result.model_dump(),
            )
