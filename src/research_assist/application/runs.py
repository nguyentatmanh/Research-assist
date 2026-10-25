"""Single-process bounded job queue, cancellation and crash recovery."""

import asyncio
import logging
from uuid import uuid4

from research_assist.application.pipeline import ResearchPipeline
from research_assist.config import Settings
from research_assist.domain.errors import CapacityError, NotFoundError, ResearchError
from research_assist.domain.models import (
    AgentEvent,
    AgentSession,
    ResearchRequest,
    ResearchResult,
    RunRecord,
    RuntimeTask,
)
from research_assist.infrastructure.repository import Repository, now

logger = logging.getLogger(__name__)


class RunManager:
    def __init__(
        self, repository: Repository, pipeline: ResearchPipeline, settings: Settings
    ) -> None:
        self.repository, self.pipeline, self.settings = repository, pipeline, settings
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_runs)

    @property
    def busy(self) -> bool:
        return bool(self.tasks)

    def submit(self, request: ResearchRequest) -> RunRecord:
        # No await between capacity check and task registration: atomic on this loop.
        if len(self.tasks) >= self.settings.max_pending_runs:
            raise CapacityError("Hàng đợi đang đầy. Hãy chờ một phiên nghiên cứu hoàn thành.")
        documents = self.repository.documents()
        if not documents:
            raise ResearchError("Hãy tải tài liệu hoặc nạp bộ dữ liệu mẫu trước khi nghiên cứu.")
        available = {document.id: document for document in documents}
        try:
            scope_documents = (
                [available[document_id] for document_id in request.document_ids]
                if request.document_ids
                else documents
            )
        except KeyError:
            raise NotFoundError("Không tìm thấy tài liệu.") from None
        # Resolve all documents once. Later uploads never alter an in-flight scope.
        request = ResearchRequest.model_validate(
            {**request.model_dump(), "document_ids": [d.id for d in scope_documents]}
        )
        run = RunRecord(
            id=f"run_{uuid4().hex}",
            status="queued",
            scope_documents=scope_documents,
            corpus_revision=self.repository.corpus_revision(),
            request=request,
            created_at=now(),
            updated_at=now(),
        )
        self.repository.save_run(run)
        self.tasks[run.id] = asyncio.create_task(self._execute(run), name=run.id)
        return run

    async def _execute(self, run: RunRecord) -> None:
        def checkpoint(result: ResearchResult) -> None:
            run.result, run.updated_at = result, now()
            self.repository.save_run(run)

        def emit(event: AgentEvent) -> None:
            run.events.append(event)
            run.updated_at = now()
            self.repository.save_run(run)

        try:
            async with self.semaphore:
                run.status, run.updated_at = "running", now()
                self.repository.save_run(run)
                async with asyncio.timeout(self.settings.run_timeout_seconds):
                    run.result = await self.pipeline.run(run.id, run.request, emit, checkpoint)
                run.status = "completed"
        except asyncio.CancelledError:
            run.status = "cancelled"
            run.error = "Phiên nghiên cứu đã được dừng."
            if run.result is not None:
                run.result.stop_reason = "cancelled"
                run.result.warnings.append(run.error)
            # Cancellation is persisted, including for queued tasks.
        except TimeoutError:
            run.status, run.error = (
                "failed",
                "Phiên nghiên cứu vượt thời gian tối đa. Thử giảm top_k hoặc đổi model.",
            )
            if run.result is not None and run.result.reported_claim_ids:
                run.status = "completed"
                run.result = run.result.model_copy(
                    update={
                        "stop_reason": "timeout",
                        "warnings": [*run.result.warnings, run.error],
                    }
                )
                emit(
                    AgentEvent(
                        sequence=len(run.events) + 1,
                        agent="orchestrator",
                        status="completed",
                        round=run.events[-1].round if run.events else 1,
                        message="Hết thời gian; đã giữ báo cáo được kiểm chứng gần nhất.",
                        details={"stop_reason": "timeout"},
                        created_at=now(),
                    )
                )
        except ResearchError as exc:
            run.status, run.error = "failed", str(exc)
        except Exception as exc:
            run.status, run.error = "failed", "Có lỗi nội bộ. Xem mã phiên để kiểm tra và chạy lại."
            # Log the exception type only; messages may contain input or credentials.
            logger.error("Research run %s failed (%s)", run.id, type(exc).__name__)
        finally:
            run.updated_at = now()
            if run.status in {"failed", "cancelled"}:
                emit(
                    AgentEvent(
                        sequence=len(run.events) + 1,
                        agent="orchestrator",
                        status="cancelled" if run.status == "cancelled" else "failed",
                        round=run.events[-1].round if run.events else 1,
                        message=run.error or "Phiên không hoàn thành.",
                        created_at=now(),
                    )
                )
            if run.result is not None:
                if run.status in {"failed", "cancelled"} or run.result.stop_reason == "timeout":
                    interrupted: list[RuntimeTask | AgentSession] = [
                        *run.result.task_graph,
                        *run.result.agent_sessions,
                    ]
                    for item in interrupted:
                        if item.status in {"pending", "running", "waiting"}:
                            item.status = "blocked"
                run.result.events = list(run.events)
            self.repository.save_run(run)
            self.tasks.pop(run.id, None)

    async def cancel(self, run_id: str) -> RunRecord:
        self.repository.run(run_id)
        task = self.tasks.get(run_id)
        if task:
            # Give a newly submitted task a chance to enter its try/finally before cancelling it.
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return self.repository.run(run_id)

    async def close(self) -> None:
        tasks = list(self.tasks.values())
        await asyncio.sleep(0)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
