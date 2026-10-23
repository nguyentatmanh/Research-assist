"""A per-run dependency graph with validated IDs, cycle detection and suspension."""

from research_assist.domain.errors import ActionRejected
from research_assist.domain.models import RuntimeTask, TaskRequest, TaskRequester


class TaskGraph:
    def __init__(self, tasks: dict[str, RuntimeTask], max_tasks: int) -> None:
        self.tasks, self.max_tasks = tasks, max_tasks

    def ready(self) -> list[RuntimeTask]:
        return [
            t
            for t in self.tasks.values()
            if t.status in {"pending", "waiting"}
            and all(self.tasks[dep].status == "completed" for dep in t.depends_on)
        ]

    def add(
        self,
        request: TaskRequest,
        requested_by: TaskRequester,
        parent_id: str | None = None,
    ) -> RuntimeTask:
        if len(self.tasks) >= self.max_tasks:
            raise ActionRejected("Task graph reached its task limit.")
        if any(dep not in self.tasks for dep in request.depends_on):
            raise ActionRejected("Task dependency does not exist.")
        if len(set(request.depends_on)) != len(request.depends_on):
            raise ActionRejected("Task dependencies must be unique.")
        if parent_id is not None and parent_id not in self.tasks:
            raise ActionRejected("Parent task does not exist.")
        if parent_id is not None and self.tasks[parent_id].status not in {
            "pending",
            "running",
            "waiting",
        }:
            raise ActionRejected("A completed or blocked parent cannot request more work.")
        task = RuntimeTask(
            **request.model_dump(),
            id=f"job_{len(self.tasks) + 1}",
            requested_by=requested_by,
            parent_id=parent_id,
        )
        edges = {tid: list(t.depends_on) for tid, t in self.tasks.items()}
        edges[task.id] = list(task.depends_on)
        if parent_id is not None:
            edges[parent_id].append(task.id)

        visited: set[str] = set()

        def visit(tid: str, path: set[str]) -> None:
            if tid in path:
                raise ActionRejected("Task dependency cycle is not allowed.")
            if tid in visited:
                return
            for dep in edges[tid]:
                visit(dep, path | {tid})
            visited.add(tid)

        for tid in edges:
            visit(tid, set())
        self.tasks[task.id] = task
        if parent_id is not None:
            parent = self.tasks[parent_id]
            parent.depends_on.append(task.id)
            parent.status = "waiting"
        return task

    def settle(self) -> None:
        """Propagate blocked children; never run a parent on a failed prerequisite."""
        changed = True
        while changed:
            changed = False
            for task in self.tasks.values():
                if task.status not in {"pending", "waiting"}:
                    continue
                if any(
                    self.tasks[dep].status in {"blocked", "cancelled"} for dep in task.depends_on
                ):
                    task.status, task.message = "blocked", "A prerequisite did not complete."
                    changed = True
