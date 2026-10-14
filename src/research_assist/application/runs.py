"""Single-process bounded job queue, cancellation and crash recovery."""

import asyncio
import logging
from uuid import uuid4

from research_assist.application.pipeline import ResearchPipeline
from research_assist.config import Settings
from research_assist.domain.errors import CapacityError, ResearchError
from research_assist.domain.models import AgentEvent, ResearchRequest, RunRecord
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
        if not self.repository.documents():
            raise ResearchError("Hãy tải tài liệu hoặc nạp bộ dữ liệu mẫu trước khi nghiên cứu.")
        for document_id in request.document_ids:
            self.repository.document(document_id)
        run = RunRecord(
            id=f"run_{uuid4().hex}",
            status="queued",
            request=request,
            created_at=now(),
            updated_at=now(),
        )
        self.repository.save_run(run)
        self.tasks[run.id] = asyncio.create_task(self._execute(run), name=run.id)
        return run

    async def _execute(self, run: RunRecord) -> None:
        def emit(event: AgentEvent) -> None:
            run.events.append(event)
            run.updated_at = now()
            self.repository.save_run(run)

        try:
            async with self.semaphore:
                run.status, run.updated_at = "running", now()
                self.repository.save_run(run)
                async with asyncio.timeout(self.settings.run_timeout_seconds):
                    run.result = await self.pipeline.run(run.id, run.request, emit)
                run.status = "completed"
        except asyncio.CancelledError:
            run.status = "cancelled"
            run.error = "Phiên nghiên cứu đã được dừng."
            # Cancellation is persisted, including for queued tasks.
        except TimeoutError:
            run.status, run.error = (
                "failed",
                "Phiên nghiên cứu vượt thời gian tối đa. Thử giảm top_k hoặc đổi model.",
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
