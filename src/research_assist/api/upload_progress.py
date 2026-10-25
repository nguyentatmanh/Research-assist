"""Short-lived, per-process status for a document upload in progress."""

import math
import time
from dataclasses import dataclass, field


@dataclass
class UploadRecord:
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    stage: str = "reading"
    size_bytes: int = 0
    pages: int | None = None
    text_chars: int | None = None
    chunks: int | None = None
    embedded: int = 0
    embedding_started_at: float | None = None
    last_batch_at: float | None = None
    last_batch_count: int = 0
    embedding_rate: float | None = None
    model_loading: bool = False
    error: str | None = None

    def snapshot(self) -> dict[str, object]:
        elapsed = (self.finished_at or time.monotonic()) - self.started_at
        remaining: int | None = None
        if self.stage == "embedding" and self.embedding_rate:
            if self.chunks and self.embedded < self.chunks:
                remaining = math.ceil((self.chunks - self.embedded) / self.embedding_rate)
        return {
            "stage": self.stage,
            "size_bytes": self.size_bytes,
            "pages": self.pages,
            "text_chars": self.text_chars,
            "chunks": self.chunks,
            "embedded": self.embedded,
            "embedding_percent": (
                round(self.embedded / self.chunks * 100)
                if self.chunks
                and self.embedding_started_at is not None
                and self.stage in {"embedding", "saving", "completed"}
                else None
            ),
            "elapsed_seconds": math.floor(elapsed),
            "remaining_seconds": remaining,
            "embedding_rate": round(self.embedding_rate, 3) if self.embedding_rate else None,
            "model_loading": self.model_loading,
            "error": self.error,
        }


class UploadProgress:
    def __init__(self) -> None:
        self.records: dict[str, UploadRecord] = {}

    def start(self, upload_id: str, size_bytes: int) -> None:
        # Keep completed records briefly for a final UI poll without unbounded memory.
        if len(self.records) >= 128:
            finished = [
                key
                for key, record in self.records.items()
                if record.stage in {"completed", "failed"}
            ]
            for key in finished[: max(1, len(self.records) - 127)]:
                self.records.pop(key, None)
        self.records[upload_id] = UploadRecord(size_bytes=size_bytes)

    def update(self, upload_id: str, stage: str, values: dict[str, int]) -> None:
        record = self.records[upload_id]
        record.stage = stage
        if stage == "embedding" and "embedded" in values:
            now = time.monotonic()
            if record.last_batch_at is not None and values["embedded"] > record.last_batch_count:
                seconds = max(now - record.last_batch_at, 0.001)
                record.embedding_rate = (values["embedded"] - record.last_batch_count) / seconds
            record.last_batch_at = now
            record.last_batch_count = values["embedded"]
        for key in ("pages", "text_chars", "chunks", "embedded"):
            if key in values:
                setattr(record, key, values[key])
        if "model_loading" in values:
            record.model_loading = bool(values["model_loading"])
        if stage == "embedding" and record.embedding_started_at is None:
            record.embedding_started_at = time.monotonic()

    def finish(self, upload_id: str, error: str | None = None) -> None:
        record = self.records[upload_id]
        record.stage = "failed" if error else "completed"
        record.finished_at = time.monotonic()
        record.error = error

    def get(self, upload_id: str) -> dict[str, object] | None:
        record = self.records.get(upload_id)
        return record.snapshot() if record else None
