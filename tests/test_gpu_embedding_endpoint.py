"""Exercise standalone notebook serving code and the app adapter without a GPU/network."""

import ast
import json
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest
from pydantic import SecretStr, ValidationError

from notebooks.gpu_embedding_server import (
    checkpoint_identity,
    create_embedding_app,
    extract_checkpoint,
)
from research_assist.application.embedding_settings import RemoteEmbeddingInput, probe_endpoint
from research_assist.config import Settings
from research_assist.domain.errors import ProviderError
from research_assist.infrastructure.embeddings import RemoteEmbeddings

ROOT = Path(__file__).parents[1]
KEY = "test-only-key-" + "x" * 32
MODEL = "research-e5-" + "a" * 32
QWEN_MODEL = "research-qwen3-" + "b" * 32


class Encoder:
    def __init__(self):
        self.inputs = []

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        self.inputs.extend(texts)
        return np.asarray([[0.6, 0.8] for _ in texts])


class QwenEncoder(Encoder):
    def encode_query(self, texts, **kwargs):
        self.inputs.extend(("query", text) for text in texts)
        return np.asarray([[0.6, 0.8] for _ in texts])

    def encode_document(self, texts, **kwargs):
        self.inputs.extend(("passage", text) for text in texts)
        return np.asarray([[0.6, 0.8] for _ in texts])


def settings(**overrides):
    return Settings(
        _env_file=None,
        embedding_provider="remote",
        embedding_model=MODEL,
        embedding_api_key=SecretStr(KEY),
        embedding_base_url="https://gpu.test/v1",
        **overrides,
    )


@pytest.mark.parametrize(
    "name", ["03_kaggle_e5_train_and_serve.ipynb", "04_gpu_e5_inference_cloudflare.ipynb"]
)
def test_notebooks_standalone_unexecuted_with_guarded_serving(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len({c["id"] for c in notebook["cells"]}) == len(notebook["cells"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            assert cell["execution_count"] is None and cell["outputs"] == []
            assert "from research_assist" not in source
            if "def create_embedding_app" in source:
                scope = {"_RESEARCHDESK_ENV_READY": False}
                exec(compile(source, name, "exec"), scope)
                assert "create_embedding_app" not in scope


def test_train_and_serve_keeps_existing_training_and_disables_server_by_default():
    original = json.loads((ROOT / "notebooks/01_kaggle_finetune_e5.ipynb").read_text("utf-8"))
    book = json.loads((ROOT / "notebooks/03_kaggle_e5_train_and_serve.ipynb").read_text("utf-8"))
    assert [c["source"] for c in book["cells"][1 : 1 + len(original["cells"])]] == [
        c["source"] for c in original["cells"]
    ]
    config = next(
        "".join(c["source"])
        for c in book["cells"]
        if "START_GPU_SERVER = False" in "".join(c["source"])
    )
    assert "START_CLOUDFLARE = False" in config


async def test_remote_adapter_preserves_e5_intent_and_checkpoint_identity_across_urls():
    encoder = Encoder()
    app = create_embedding_app(encoder, KEY, MODEL)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://gpu.test"
    ) as client:
        remote = RemoteEmbeddings(settings(), client)
        assert await remote.embed(["câu hỏi"], query=True) == [[0.6, 0.8]]
        assert await remote.embed(["đoạn tài liệu"]) == [[0.6, 0.8]]
        assert encoder.inputs == ["query: câu hỏi", "passage: đoạn tài liệu"]
        changed_url = settings().model_copy(update={"embedding_base_url": "https://other.test"})
        assert RemoteEmbeddings(changed_url, client).fingerprint == remote.fingerprint
        changed_model = settings().model_copy(update={"embedding_model": "research-e5-new"})
        assert RemoteEmbeddings(changed_model, client).fingerprint != remote.fingerprint


async def test_qwen_server_and_adapter_use_query_document_methods():
    encoder = QwenEncoder()
    app = create_embedding_app(encoder, KEY, QWEN_MODEL, family="qwen3")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://gpu.test"
    ) as client:
        qwen_settings = settings().model_copy(update={"embedding_model": QWEN_MODEL})
        remote = RemoteEmbeddings(qwen_settings, client)
        assert await remote.embed(["câu hỏi"], query=True) == [[0.6, 0.8]]
        assert await remote.embed(["tài liệu"]) == [[0.6, 0.8]]
        assert encoder.inputs == [("query", "câu hỏi"), ("passage", "tài liệu")]
        assert (await client.get("/v1/info", headers={"Authorization": "Bearer " + KEY})).json()[
            "family"
        ] == "qwen3"


async def test_qwen_family_selection_is_checked_before_saving():
    app = create_embedding_app(QwenEncoder(), KEY, QWEN_MODEL, family="qwen3")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://gpu.test"
    ) as client:
        current = Settings(_env_file=None)
        body = RemoteEmbeddingInput(
            base_url="https://gpu.test",
            api_key=KEY,
            model_family="qwen3",
            tokenizer_model="Qwen/Qwen3-Embedding-4B",
        )
        candidate, info = await probe_endpoint(body, current, client)
        assert candidate.embedding_model == QWEN_MODEL
        assert info["model_family"] == "qwen3"
        with pytest.raises(Exception, match="không khớp"):
            await probe_endpoint(body.model_copy(update={"model_family": "e5"}), current, client)


def test_qwen_notebook_is_standalone_and_serving_opt_in():
    book = json.loads(
        (ROOT / "notebooks/05_gpu_qwen3_inference_cloudflare.ipynb").read_text("utf-8")
    )
    assert book["nbformat"] == 4
    assert len({c["id"] for c in book["cells"]}) == len(book["cells"])
    source = "\n".join("".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code")
    assert "START_GPU_SERVER = False" in source
    assert "START_CLOUDFLARE = False" in source
    assert "load_qwen_checkpoint" in source
    assert 'family="qwen3"' in source
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))


async def test_server_authentication_model_binding_and_request_limits():
    app = create_embedding_app(Encoder(), KEY, MODEL)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://gpu.test"
    ) as client:
        assert (await client.get("/v1/info")).status_code == 401
        headers = {"Authorization": "Bearer " + KEY}
        assert (await client.get("/v1/info", headers=headers)).json()["model"] == MODEL
        base = {"model": MODEL, "input": ["text"], "input_type": "query"}
        for update, expected in [
            ({"model": "other"}, 409),
            ({"input": []}, 422),
            ({"input": ["x"] * 33}, 422),
            ({"input_type": "wrong"}, 422),
            ({"input": ["x" * 16001]}, 422),
        ]:
            assert (
                await client.post("/v1/embeddings", headers=headers, json={**base, **update})
            ).status_code == expected
        assert (
            await client.post("/v1/embeddings", headers=headers, content=b"x" * 300001)
        ).status_code == 413


async def test_remote_refuses_checkpoint_and_length_mismatch():
    app = create_embedding_app(Encoder(), KEY, MODEL)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://gpu.test"
    ) as client:
        remote = RemoteEmbeddings(settings(embedding_max_seq_length=256), client)
        with pytest.raises(ProviderError, match="checkpoint/config"):
            await remote.embed(["test"])
        remote = RemoteEmbeddings(
            settings().model_copy(update={"embedding_model": "research-e5-new"}), client
        )
        with pytest.raises(ProviderError, match="checkpoint/config"):
            await remote.embed(["test"])


def test_checkpoint_signature_follows_content_not_directory_name(tmp_path):
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "model.safetensors").write_bytes(b"weights")
        (folder / "modules.json").write_text("[]")
    assert checkpoint_identity(tmp_path / "a") == checkpoint_identity(tmp_path / "b")
    (tmp_path / "b" / "model.safetensors").write_bytes(b"different weights")
    assert checkpoint_identity(tmp_path / "a") != checkpoint_identity(tmp_path / "b")
    assert checkpoint_identity(tmp_path / "a", 256) != checkpoint_identity(tmp_path / "a", 512)


def test_checkpoint_extraction_rejects_traversal_and_loads_packaged_root(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_checkpoint(archive, tmp_path / "unpacked")
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("checkpoint/modules.json", "[]")
        zipped.writestr("checkpoint/model.safetensors", "weights")
    folder = extract_checkpoint(archive, tmp_path / "good")
    assert folder == tmp_path / "good" / "checkpoint"


def test_remote_requires_https_secret_and_explicit_model_id():
    with pytest.raises(ValidationError, match="HTTPS"):
        Settings(
            _env_file=None,
            embedding_provider="remote",
            embedding_model=MODEL,
            embedding_api_key=SecretStr(KEY),
            embedding_base_url="http://public.example/v1",
        )
    with pytest.raises(ValidationError, match="TOKENIZER"):
        settings(chunk_strategy="tokens")


async def test_remote_ingest_uses_separate_local_tokenizer_not_remote_checkpoint_id(
    tmp_path, monkeypatch
):
    import re

    from research_assist.application.documents import DocumentService
    from research_assist.infrastructure.repository import Repository

    seen = []

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}

    def tokenizer(model, revision):
        seen.append((model, revision))
        return Tokenizer()

    monkeypatch.setattr("research_assist.infrastructure.token_chunker.tokenizer_for", tokenizer)
    cfg = settings(
        data_dir=tmp_path,
        chunk_strategy="tokens",
        chunk_tokenizer_model="intfloat/multilingual-e5-base",
    )
    encoder = Encoder()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(create_embedding_app(encoder, KEY, MODEL)),
        base_url="https://gpu.test",
    ) as client:
        repo = Repository(tmp_path)
        doc, _ = await DocumentService(repo, RemoteEmbeddings(cfg, client), cfg).ingest(
            b"Research evidence is retrieved from original passages with exact citations.",
            "paper.txt",
        )
        assert repo.chunks([doc.id])[0].vector_model.startswith("remote:")
        assert seen == [("intfloat/multilingual-e5-base", "")]
        assert encoder.inputs[0].startswith("passage: ")


async def test_remote_rejects_invalid_vector_shapes_and_never_silently_falls_back():
    def provider(request):
        if request.url.path.endswith("/info"):
            return httpx.Response(
                200,
                json={
                    "model": MODEL,
                    "family": "e5",
                    "normalized": True,
                    "protocol": "research-embedding-v1",
                    "max_seq_length": 512,
                },
            )
        return httpx.Response(
            200, json={"model": MODEL, "data": [{"index": 0, "embedding": [0, 0]}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        with pytest.raises(ProviderError):
            await RemoteEmbeddings(settings(), client).embed(["test"])
