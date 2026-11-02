"""Short atomic commits around task-scoped, cancellable asynchronous execution."""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from research_assist.domain.errors import ActionRejected
from research_assist.domain.models import ResearchTaskState, RuntimeTask

T = TypeVar("T")


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


@dataclass(frozen=True)
class TaskExecutionContext:
    task_id: str
    task_generation: int
    dependency_fingerprint: str
    allowed_chunk_ids: tuple[str, ...]
    requirement_ids: tuple[str, ...]
    idempotency_key: str


class TaskExecutor:
    def __init__(self) -> None:
        self.commit_lock = asyncio.Lock()
        self.active = True

    def prepare(
        self, task: RuntimeTask, state: ResearchTaskState, dependency: str
    ) -> TaskExecutionContext:
        if not self.is_running(task):
            raise ActionRejected("Task execution is no longer active.")
        return TaskExecutionContext(
            task.id,
            task.generation,
            dependency,
            state.scope.allowed_chunk_ids,
            state.scope.requirement_ids,
            fingerprint([task.id, task.generation, task.attempts, dependency]),
        )

    def is_running(self, task: RuntimeTask) -> bool:
        return self.active and task.status == "running"

    async def execute(
        self,
        task: RuntimeTask,
        state: ResearchTaskState,
        dependency: Callable[[], str],
        work: Callable[[TaskExecutionContext], Awaitable[T]],
        publish: Callable[[T], None],
    ) -> T:
        context = self.prepare(task, state, dependency())
        # Work owns its input snapshot and never holds the commit lock over I/O.
        value = await work(context)
        async with self.commit_lock:
            if (
                not self.active
                or task.status != "running"
                or task.generation != context.task_generation
                or dependency() != context.dependency_fingerprint
            ):
                raise ActionRejected("Task lease or dependency changed before commit.")
            if context.idempotency_key not in state.committed_actions:
                publish(value)
                state.committed_actions.append(context.idempotency_key)
                state.source_fingerprint = context.dependency_fingerprint
            task.status = "completed"
        return value

    def stop(self, tasks: list[RuntimeTask]) -> None:
        self.active = False
        for task in tasks:
            if task.status in {"pending", "waiting", "retry_waiting", "running"}:
                task.generation += 1
                task.status = "cancelled"
                task.message = "Research run stopped before this task completed."
