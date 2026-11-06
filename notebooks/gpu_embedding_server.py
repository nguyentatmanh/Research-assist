"""Authenticated, bounded E5 inference; usable on a permitted notebook or GPU host."""

import asyncio
import hashlib
import json
import secrets
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError


def checkpoint_identity(path, max_seq_length=512, family="e5"):
    """Identity follows weights/tokenizer/config, never the temporary tunnel URL."""
    root = Path(path)
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix
        in {
            ".safetensors",
            ".bin",
            ".json",
            ".model",
            ".txt",
        }
        and "checkpoints" not in p.relative_to(root).parts
    )
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise ValueError("Checkpoint must contain model weights and tokenizer/config files.")
    if family not in {"e5", "qwen3"}:
        raise ValueError("Unsupported embedding family.")
    digest = hashlib.sha256(f"{family}:query:passage:normalized:v1:{max_seq_length}".encode())
    for file in files:
        digest.update(str(file.relative_to(root)).replace("\\", "/").encode())
        with file.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return f"research-{family}-" + digest.hexdigest()[:32]


class EmbeddingInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    input: str | list[str]
    input_type: Literal["query", "passage"]


def create_embedding_app(
    encoder, api_key, model_id, max_seq_length=512, family="e5", tokenizer_model=""
):
    if len(api_key) < 32:
        raise ValueError("Use an API key with at least 32 characters.")
    if family not in {"e5", "qwen3"} or not model_id.startswith(f"research-{family}-"):
        raise ValueError("Model ID must match the embedding family.")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    lock = threading.Lock()
    active = 0
    dimension = int(encoder.get_sentence_embedding_dimension())

    def authorize(request):
        header = request.headers.get("authorization", "")
        if not secrets.compare_digest(header, "Bearer " + api_key):
            raise HTTPException(401, "Invalid API key.")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/info")
    async def info(request: Request):
        authorize(request)
        return {
            "model": model_id,
            "family": family,
            "tokenizer_model": tokenizer_model,
            "dimension": dimension,
            "max_seq_length": max_seq_length,
            "normalized": True,
            "protocol": "research-embedding-v1",
            "max_batch_size": 32,
        }

    def encode(texts, input_type):
        with lock:
            if family == "e5":
                method = encoder.encode
                inputs = [("query: " if input_type == "query" else "passage: ") + t for t in texts]
            else:
                method = encoder.encode_query if input_type == "query" else encoder.encode_document
                inputs = texts
            vectors = method(
                inputs,
                batch_size=8 if family == "e5" else 2,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return vectors.tolist()

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        nonlocal active
        authorize(request)
        if active >= 4:
            raise HTTPException(429, "GPU queue is full; retry later.")
        body = bytearray()
        async for part in request.stream():
            body.extend(part)
            if len(body) > 300000:
                raise HTTPException(413, "Request is too large.")
        try:
            value = EmbeddingInput.model_validate_json(body)
        except (ValidationError, ValueError):
            raise HTTPException(422, "Invalid embedding request.") from None
        if value.model != model_id:
            raise HTTPException(409, "Checkpoint changed; configure the new model ID and reindex.")
        texts = [value.input] if isinstance(value.input, str) else value.input
        if not 1 <= len(texts) <= 32 or any(
            not text.strip() or len(text) > 16000 for text in texts
        ):
            raise HTTPException(422, "Use 1-32 nonempty texts, at most 16000 characters each.")
        if active >= 4:
            raise HTTPException(429, "GPU queue is full; retry later.")
        # Retain the queue slot and physical lock until the thread actually exits.
        active += 1
        job = asyncio.create_task(asyncio.to_thread(encode, texts, value.input_type))

        def finished(task):
            nonlocal active
            active -= 1
            if not task.cancelled():
                task.exception()  # Retrieve a late exception after an HTTP cancellation.

        job.add_done_callback(finished)
        try:
            vectors = await asyncio.shield(job)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise HTTPException(
                503, "GPU inference failed; inspect the private server log."
            ) from None
        return {
            "object": "list",
            "model": model_id,
            "data": [
                {"object": "embedding", "index": i, "embedding": vector}
                for i, vector in enumerate(vectors)
            ],
        }

    return app


def extract_checkpoint(archive, output):
    """Extract a downloaded training checkpoint; reject traversal and symlinks."""
    import zipfile

    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Choose an empty checkpoint extraction directory.")
    with zipfile.ZipFile(archive) as zipped:
        total = 0
        for item in zipped.infolist():
            target = (root / item.filename.replace("\\", "/")).resolve()
            if not target.is_relative_to(root) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe checkpoint archive entry.")
            total += item.file_size
            if total > 16 * 1024**3:
                raise ValueError("Checkpoint archive exceeds 16 GiB.")
        zipped.extractall(root)
    candidates = list(root.rglob("modules.json"))
    if len(candidates) != 1:
        raise ValueError("Use the checkpoint ZIP, not the full experiment/checkpoints ZIP.")
    return candidates[0].parent


def load_e5_checkpoint(path, device="cuda", max_seq_length=512):
    from sentence_transformers import SentenceTransformer

    root = Path(path)
    manifest = json.loads((root / "research_model.json").read_text(encoding="utf-8"))
    if manifest.get("model_family") != "e5" or not manifest.get("fine_tuned"):
        raise ValueError("Select an E5 checkpoint with actual optimizer steps (fine_tuned=true).")
    model = SentenceTransformer(
        str(root), device=device, local_files_only=True, trust_remote_code=False
    )
    model.max_seq_length = max_seq_length
    return model, checkpoint_identity(root, max_seq_length)


def load_qwen_checkpoint(path, device="cuda", max_seq_length=512):
    """Load a local Qwen3 embedding snapshot; never substitute a base model silently."""
    import torch
    from sentence_transformers import SentenceTransformer

    root = Path(path)
    if not root.is_dir():
        raise ValueError("Select a local Qwen3-Embedding checkpoint directory.")
    manifest_path = root / "research_model.json"
    if (
        manifest_path.is_file()
        and json.loads(manifest_path.read_text("utf-8")).get("format") == "swift_qwen3_adapter_v1"
    ):
        loader = globals().get("load_swift_qwen_adapter")
        if loader is None:
            from notebooks.qwen_adapter_encoder import load_swift_qwen_adapter

            loader = load_swift_qwen_adapter
        return loader(root, max_seq_length)
    config_files = list(root.glob("config.json")) or list(root.glob("0_Transformer/config.json"))
    if (
        not config_files
        or json.loads(config_files[0].read_text(encoding="utf-8")).get("model_type") != "qwen3"
    ):
        raise ValueError("Checkpoint is not a Qwen3 embedding model.")
    model = SentenceTransformer(
        str(root),
        device=device,
        local_files_only=True,
        trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.float16},
    )
    model.max_seq_length = max_seq_length
    if not hasattr(model, "encode_query") or not hasattr(model, "encode_document"):
        raise ValueError("Checkpoint does not support query/document embedding.")
    return model, checkpoint_identity(root, max_seq_length, family="qwen3")
