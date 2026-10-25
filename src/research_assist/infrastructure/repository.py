"""SQLite persistence. Each operation owns a connection and all writes are transactional."""

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from research_assist.domain.errors import DocumentError, NotFoundError
from research_assist.domain.models import Document, IndexedChunk, RunRecord


def now() -> str:
    return datetime.now(UTC).isoformat()


class Repository:
    def __init__(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "research.sqlite3"
        self.blobs = data_dir / "blobs"
        self.blobs.mkdir(exist_ok=True)
        with self.connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2, 3):
                raise RuntimeError("Unsupported database version; back up data before upgrading")
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL,
                    metadata TEXT NOT NULL, blob_name TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS generation_cache (
                    key TEXT PRIMARY KEY, payload TEXT NOT NULL, expires REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS retrieval_revision (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1), revision INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO retrieval_revision VALUES (1, 0);
                CREATE TRIGGER IF NOT EXISTS retrieval_on_insert AFTER INSERT ON chunks BEGIN
                    UPDATE retrieval_revision SET revision=revision+1 WHERE singleton=1;
                END;
                CREATE TRIGGER IF NOT EXISTS retrieval_on_update AFTER UPDATE ON chunks BEGIN
                    UPDATE retrieval_revision SET revision=revision+1 WHERE singleton=1;
                END;
                CREATE TRIGGER IF NOT EXISTS retrieval_on_delete AFTER DELETE ON chunks BEGIN
                    UPDATE retrieval_revision SET revision=revision+1 WHERE singleton=1;
                END;
                PRAGMA user_version=3;
            """)

    def cache_get(self, key: str) -> str | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT payload FROM generation_cache WHERE key=? AND expires>?", (key, time.time())
            ).fetchone()
        return str(row["payload"]) if row else None

    def cache_put(self, key: str, payload: str, ttl: int, max_entries: int) -> None:
        with self.connection() as db:
            db.execute("DELETE FROM generation_cache WHERE expires<=?", (time.time(),))
            db.execute(
                "INSERT OR REPLACE INTO generation_cache VALUES (?,?,?)",
                (key, payload, time.time() + ttl),
            )
            db.execute(
                "DELETE FROM generation_cache WHERE key IN "
                "(SELECT key FROM generation_cache ORDER BY expires DESC LIMIT -1 OFFSET ?)",
                (max_entries,),
            )

    def clear_cache(self) -> int:
        with self.connection() as db:
            return db.execute("DELETE FROM generation_cache").rowcount

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def documents(self) -> list[Document]:
        with self.connection() as db:
            rows = db.execute("SELECT metadata FROM documents ORDER BY rowid DESC").fetchall()
        return [Document.model_validate_json(row["metadata"]) for row in rows]

    def document(self, document_id: str) -> Document:
        with self.connection() as db:
            row = db.execute("SELECT metadata FROM documents WHERE id=?", (document_id,)).fetchone()
        if row is None:
            raise NotFoundError("Không tìm thấy tài liệu.")
        return Document.model_validate_json(row["metadata"])

    def by_hash(self, digest: str) -> Document | None:
        with self.connection() as db:
            row = db.execute("SELECT metadata FROM documents WHERE sha256=?", (digest,)).fetchone()
        return Document.model_validate_json(row["metadata"]) if row else None

    def add_document(self, document: Document, chunks: list[IndexedChunk], blob_name: str) -> None:
        with self.connection() as db:
            db.execute(
                "INSERT INTO documents VALUES (?,?,?,?)",
                (
                    document.id,
                    document.sha256,
                    document.model_dump_json(),
                    blob_name,
                ),
            )
            db.executemany(
                "INSERT INTO chunks VALUES (?,?,?)",
                [(c.id, c.document_id, c.model_dump_json()) for c in chunks],
            )

    def chunks(self, document_ids: list[str] | None = None) -> list[IndexedChunk]:
        sql, args = "SELECT payload FROM chunks", []
        if document_ids:
            sql += " WHERE document_id IN (" + ",".join("?" for _ in document_ids) + ")"
            args = document_ids
        sql += " ORDER BY rowid"
        with self.connection() as db:
            rows = db.execute(sql, args).fetchall()
        return [IndexedChunk.model_validate_json(row["payload"]) for row in rows]

    def corpus_revision(self) -> int:
        with self.connection() as db:
            return int(
                db.execute("SELECT revision FROM retrieval_revision WHERE singleton=1").fetchone()[
                    0
                ]
            )

    def retrieval_snapshot(self, document_ids: list[str]) -> tuple[int, list[IndexedChunk]]:
        # Revision and rows come from ONE SQLite read transaction, including other writers.
        with self.connection() as db:
            db.execute("BEGIN")
            revision = int(
                db.execute("SELECT revision FROM retrieval_revision WHERE singleton=1").fetchone()[
                    0
                ]
            )
            sql = "SELECT payload FROM chunks"
            if document_ids:
                sql += " WHERE document_id IN (" + ",".join("?" for _ in document_ids) + ")"
            rows = db.execute(sql + " ORDER BY rowid", document_ids).fetchall()
        return revision, [IndexedChunk.model_validate_json(row["payload"]) for row in rows]

    def chunk_count(self) -> int:
        with self.connection() as db:
            return int(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def update_vectors(self, chunks: list[IndexedChunk]) -> None:
        with self.connection() as db:
            db.executemany(
                "UPDATE chunks SET payload=? WHERE id=?",
                [(c.model_dump_json(), c.id) for c in chunks],
            )

    def replace_index(
        self,
        staged: list[tuple[Document, list[IndexedChunk]]],
        *,
        expected_revision: int | None = None,
    ) -> None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            revision = db.execute(
                "SELECT revision FROM retrieval_revision WHERE singleton=1"
            ).fetchone()[0]
            if expected_revision is not None and revision != expected_revision:
                raise DocumentError(
                    "Kho tài liệu đã thay đổi trong khi rebuild; chỉ mục mới chưa được áp dụng."
                )
            for document, chunks in staged:
                db.execute("DELETE FROM chunks WHERE document_id=?", (document.id,))
                db.executemany(
                    "INSERT INTO chunks VALUES (?,?,?)",
                    [(c.id, c.document_id, c.model_dump_json()) for c in chunks],
                )
                db.execute(
                    "UPDATE documents SET metadata=? WHERE id=?",
                    (document.model_dump_json(), document.id),
                )
            db.execute("DELETE FROM generation_cache")

    def blob(self, document_id: str) -> Path:
        with self.connection() as db:
            row = db.execute(
                "SELECT blob_name FROM documents WHERE id=?", (document_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError("Không tìm thấy tài liệu.")
        path = self.blobs / str(row["blob_name"])
        if not path.is_file():
            raise NotFoundError("Tệp gốc không còn trong kho dữ liệu.")
        return path

    def delete_document(self, document_id: str) -> None:
        path = self.blob(document_id)
        with self.connection() as db:
            db.execute("DELETE FROM documents WHERE id=?", (document_id,))
        path.unlink(missing_ok=True)
        # Completed runs contain their own evidence snapshots; original file links may expire.

    def save_run(self, run: RunRecord) -> None:
        with self.connection() as db:
            db.execute(
                """INSERT INTO runs VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                       status=excluded.status, payload=excluded.payload""",
                (
                    run.id,
                    run.status,
                    run.created_at,
                    run.model_dump_json(),
                ),
            )

    def run(self, run_id: str) -> RunRecord:
        with self.connection() as db:
            row = db.execute("SELECT payload FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise NotFoundError("Không tìm thấy phiên nghiên cứu.")
        return RunRecord.model_validate_json(row["payload"])

    def history(self, limit: int = 40) -> list[dict[str, object]]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT payload FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        history = []
        for row in rows:
            data = json.loads(row["payload"])
            history.append(
                {
                    "id": data["id"],
                    "status": data["status"],
                    "created_at": data["created_at"],
                    "question": data["request"]["question"],
                    "title": data["result"]["title"] if data.get("result") else None,
                    "stop_reason": (data.get("result") or {}).get("stop_reason"),
                    "coverage_complete": (
                        bool((data.get("result") or {}).get("coverage"))
                        and all(
                            c.get("status") == "covered"
                            for c in (data.get("result") or {}).get("coverage", [])
                        )
                    ),
                    "reported_claim_count": len(
                        (data.get("result") or {}).get("reported_claim_ids", [])
                    ),
                }
            )
        return history

    def recover_interrupted(self) -> None:
        with self.connection() as db:
            rows = db.execute(
                "SELECT payload FROM runs WHERE status IN ('running','queued')"
            ).fetchall()
            for row in rows:
                run = RunRecord.model_validate_json(row["payload"])
                run.status, run.updated_at = "failed", now()
                run.error = "Máy chủ đã khởi động lại. Hãy tạo phiên nghiên cứu mới."
                db.execute(
                    "UPDATE runs SET status=?, payload=? WHERE id=?",
                    (
                        run.status,
                        run.model_dump_json(),
                        run.id,
                    ),
                )
