import re
from pathlib import Path

from research_assist.config import Settings
from research_assist.domain.models import Document, SourceSection
from research_assist.infrastructure.documents import chunk_sections
from research_assist.infrastructure.embeddings import LocalEmbeddings


def test_token_windows_preserve_unicode_source_offsets(
    tmp_path: Path, document: Document, monkeypatch
) -> None:
    class Tokenizer:
        def __call__(self, text: str, **kwargs):
            return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}

    monkeypatch.setattr(
        "research_assist.infrastructure.token_chunker.tokenizer_for", lambda *args: Tokenizer()
    )
    text = "  " + " ".join(f"bằng_chứng_{i}" for i in range(150)) + "  "
    settings = Settings(
        data_dir=tmp_path,
        chunk_strategy="tokens",
        chunk_tokens=64,
        chunk_token_overlap=8,
        embedding_max_seq_length=128,
        _env_file=None,
    )
    section = SourceSection(text=text, page=3)
    chunks = chunk_sections([section], document, settings)
    assert len(chunks) == 3
    assert all(section.text[c.start : c.end] == c.text and c.page == 3 for c in chunks)
    assert chunks[1].start < chunks[0].end


def test_local_checkpoint_family_and_changed_weights(tmp_path: Path) -> None:
    import json

    (tmp_path / "research_model.json").write_text(json.dumps({"model_family": "e5"}))
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"before")
    before = LocalEmbeddings(str(tmp_path), 2)
    assert before.family == "e5"  # folder name contains no E5 hint
    weights.write_bytes(b"after")
    after = LocalEmbeddings(str(tmp_path), 2)
    assert before.fingerprint != after.fingerprint
