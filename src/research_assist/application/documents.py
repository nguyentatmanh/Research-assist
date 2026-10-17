import asyncio
import hashlib
from pathlib import Path
from uuid import uuid4

from research_assist.config import Settings
from research_assist.domain.errors import CapacityError, DocumentError
from research_assist.domain.metadata import SECTION_INDEX
from research_assist.domain.models import Document, IndexedChunk
from research_assist.domain.ports import Embedder
from research_assist.infrastructure.documents import chunk_sections, extract, safe_filename
from research_assist.infrastructure.repository import Repository, now


class DocumentService:
    def __init__(self, repository: Repository, embedder: Embedder, settings: Settings) -> None:
        self.repository, self.embedder, self.settings = repository, embedder, settings
        self.lock = asyncio.Lock()
        self.maintenance = False
        self.sample_dir = Path(__file__).resolve().parents[1] / "sample_data"

    async def ingest(self, data: bytes, filename: str) -> tuple[Document, bool]:
        filename = safe_filename(filename)
        async with self.lock:
            digest = hashlib.sha256(data).hexdigest()
            existing = self.repository.by_hash(digest)
            if existing:
                return existing, True
            sections = await asyncio.to_thread(extract, data, filename, self.settings)
            first_line = next((s.text.splitlines()[0] for s in sections if s.text), filename)
            title = (
                first_line.lstrip("# ")[:180] if filename.endswith(".md") else Path(filename).stem
            )
            document = Document(
                id=f"doc_{uuid4().hex}",
                filename=filename,
                title=title,
                kind=Path(filename).suffix.lower().lstrip("."),
                sha256=digest,
                page_count=len(sections) if Path(filename).suffix.lower() == ".pdf" else 0,
                chunk_count=0,
                created_at=now(),
                warnings=[message for section in sections for message in section.warnings],
                index_version=SECTION_INDEX,
            )
            chunks = await asyncio.to_thread(chunk_sections, sections, document, self.settings)
            if not chunks:
                raise DocumentError("Không tạo được đoạn văn bản từ tệp.")
            if len(chunks) + self.repository.chunk_count() > self.settings.max_chunks:
                raise CapacityError(
                    "Kho tài liệu đã đạt giới hạn chunk; xóa tài liệu cũ trước khi tải thêm."
                )
            vectors = await self.embedder.embed([chunk.text for chunk in chunks])
            indexed = [
                IndexedChunk(
                    **chunk.model_dump(),
                    vector=vectors[i] if vectors else None,
                    vector_model=self.embedder.fingerprint,
                )
                for i, chunk in enumerate(chunks)
            ]
            document.chunk_count = len(chunks)
            blob_name = document.id + "." + document.kind
            blob = self.repository.blobs / blob_name
            # No await between file creation and the small SQL transaction: cancellation cannot
            # commit the row in a background thread after removing its file.
            try:
                blob.write_bytes(data)
                self.repository.add_document(document, indexed, blob_name)
            except Exception:
                blob.unlink(missing_ok=True)
                raise
            return document, False

    async def delete(self, document_id: str) -> None:
        async with self.lock:
            self.repository.delete_document(document_id)

    async def reindex(self) -> int:
        if self.maintenance:
            raise CapacityError("Đang xây dựng lại chỉ mục.")
        self.maintenance = True
        try:
            async with self.lock:
                chunks = self.repository.chunks()
                if not chunks:
                    return 0
                vectors = await self.embedder.embed([chunk.text for chunk in chunks])
                for index, chunk in enumerate(chunks):
                    chunk.vector = vectors[index] if vectors else None
                    chunk.vector_model = self.embedder.fingerprint
                self.repository.update_vectors(chunks)
                return len(chunks)
        finally:
            self.maintenance = False

    async def rebuild(self) -> int:
        """Stage all extraction/chunks/vectors; replace in one transaction only on success."""
        if self.maintenance:
            raise CapacityError("Đang xây dựng lại chỉ mục.")
        self.maintenance = True
        try:
            async with self.lock:
                staged = []
                count = 0
                for document in self.repository.documents():
                    data = await asyncio.to_thread(self.repository.blob(document.id).read_bytes)
                    sections = await asyncio.to_thread(
                        extract, data, document.filename, self.settings
                    )
                    chunks = await asyncio.to_thread(
                        chunk_sections, sections, document, self.settings
                    )
                    count += len(chunks)
                    if not chunks or count > self.settings.max_chunks:
                        raise DocumentError(
                            "Rebuild vượt giới hạn hoặc không tạo được chunk; chỉ mục cũ được giữ."
                        )
                    vectors = await self.embedder.embed([c.text for c in chunks])
                    indexed = [
                        IndexedChunk(
                            **c.model_dump(),
                            vector=vectors[i] if vectors else None,
                            vector_model=self.embedder.fingerprint,
                        )
                        for i, c in enumerate(chunks)
                    ]
                    document = document.model_copy(
                        update={
                            "chunk_count": len(indexed),
                            "index_version": SECTION_INDEX,
                            "warnings": [w for section in sections for w in section.warnings],
                        }
                    )
                    staged.append((document, indexed))
                # No await: cancellation cannot split this commit. Old reports own source snapshots.
                self.repository.replace_index(staged)
                return count
        finally:
            self.maintenance = False

    async def seed(self) -> list[Document]:
        documents = []
        paths = await asyncio.to_thread(lambda: sorted(self.sample_dir.glob("*.md")))
        for path in paths:
            data = await asyncio.to_thread(path.read_bytes)
            document, _ = await self.ingest(data, path.name)
            documents.append(document)
        return documents
