"""Token windows preserve original character offsets for verifiable quotes."""

from functools import lru_cache
from typing import Any
from uuid import uuid4

from research_assist.config import Settings
from research_assist.domain.errors import DocumentError
from research_assist.domain.models import Chunk, Document, SourceSection


@lru_cache(maxsize=4)
def tokenizer_for(model: str, revision: str) -> Any:
    from transformers import AutoTokenizer

    options: dict[str, Any] = {"use_fast": True, "trust_remote_code": False}
    if revision:
        options["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model, **options)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for exact offsets")
    return tokenizer


def token_chunks(
    sections: list[SourceSection], document: Document, settings: Settings
) -> list[Chunk]:
    try:
        tokenizer = tokenizer_for(settings.embedding_model, settings.embedding_revision)
        chunks: list[Chunk] = []
        stride = settings.chunk_tokens - settings.chunk_token_overlap
        for section in sections:
            offsets = tokenizer(
                section.text, add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]
            for index in range(0, len(offsets), stride):
                window = offsets[index : index + settings.chunk_tokens]
                start, end = window[0][0], window[-1][1]
                raw = section.text[start:end]
                start += len(raw) - len(raw.lstrip())
                text = raw.strip()
                if len(text) >= 20:
                    chunks.append(
                        Chunk(
                            id=f"ch_{uuid4().hex}",
                            document_id=document.id,
                            document_title=document.title,
                            filename=document.filename,
                            text=text,
                            page=section.page,
                            section=section.label,
                            ordinal=len(chunks),
                            start=start,
                            end=start + len(text),
                        )
                    )
                if index + settings.chunk_tokens >= len(offsets):
                    break
        return chunks
    except Exception as exc:
        raise DocumentError(
            "Không chạy được tokenizer. Cài extra local-models và tải embedding model trước."
        ) from exc
