"""Authenticated embedding, reranking and memory telemetry for the Kaggle worker."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Annotated, Any, Literal

import torch
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from kaggle.security import LOOPBACK, require_bearer
from notebooks.gpu_embedding_server import checkpoint_identity

app = FastAPI(
    title="Research Assist Kaggle GPU Worker", docs_url=None, redoc_url=None, openapi_url=None
)
Authenticated = Annotated[None, Depends(require_bearer)]

_models: dict[str, Any] = {}
_tokenizers: dict[str, Any] = {}
_embedding_info: dict[str, Any] = {}
_quiesce_until = 0.0
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
DEV_EMBED = "cuda:1" if GPU_COUNT >= 2 else ("cuda:0" if GPU_COUNT == 1 else "cpu")
DEV_RERANK = DEV_EMBED
DEV_LLM = "cuda:0" if GPU_COUNT >= 1 else "cpu"
_locks = {device: Lock() for device in dict.fromkeys((DEV_EMBED, DEV_LLM))}
_slots = {device: BoundedSemaphore(4) for device in _locks}

EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "intfloat/multilingual-e5-small")
RERANK_MODEL_ID = os.environ.get("RERANK_MODEL_ID", "BAAI/bge-reranker-large")
GIB = 1024**3
EMBED_MAX_LENGTH = 512
EMBED_MAX_BATCH = 32


def get_vram_stats() -> dict[str, Any]:
    if not CUDA_AVAILABLE:
        return {"status": "cpu_only", "devices": []}
    devices = []
    for i in range(GPU_COUNT):
        free, total = torch.cuda.mem_get_info(i)
        devices.append(
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "vram_total_gb": round(total / GIB, 2),
                "vram_free_gb": round(free / GIB, 2),
                "vram_allocated_gb": round(torch.cuda.memory_allocated(i) / GIB, 2),
                "vram_reserved_gb": round(torch.cuda.memory_reserved(i) / GIB, 2),
                "oom_count": torch.cuda.memory_stats(i).get("num_ooms", 0),
            }
        )
    return {"status": "cuda_online", "gpu_count": GPU_COUNT, "devices": devices}


@contextmanager
def inference(device: str) -> Iterator[None]:
    slots = _slots[device]
    if not slots.acquire(blocking=False):
        raise HTTPException(429, "Hàng đợi suy luận đầy. Hãy thử lại sau.")
    try:
        with _locks[device], torch.inference_mode():
            if time.monotonic() < _quiesce_until:
                raise HTTPException(503, "GPU Worker đang dừng để bảo trì.")
            yield
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(503, "GPU hết bộ nhớ. Giảm kích thước lượt xử lý.") from None
    finally:
        slots.release()


def load_embedding_model() -> Any:
    if "embedder" in _models:
        return _models["embedder"]
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    checkpoint = Path(EMBED_MODEL_ID)
    local = checkpoint.is_dir()
    if not local:
        checkpoint = Path(
            snapshot_download(
                EMBED_MODEL_ID,
                revision=os.environ.get("EMBED_MODEL_REVISION") or None,
                allow_patterns=["*.json", "*.safetensors", "pytorch_model.bin", "*.model", "*.txt"],
                ignore_patterns=["onnx/*", "openvino/*"],
            )
        )
    model = SentenceTransformer(str(checkpoint), device=DEV_EMBED)
    max_length = min(EMBED_MAX_LENGTH, model.max_seq_length)
    model.max_seq_length = max_length
    if local:
        identity = checkpoint_identity(checkpoint, max_seq_length=max_length, family="e5")
    else:
        key = f"{EMBED_MODEL_ID}:{checkpoint.name}:e5:normalized:v1:{max_length}"
        identity = "research-e5-" + hashlib.sha256(key.encode()).hexdigest()[:32]
    _embedding_info.update(
        model=identity,
        family="e5",
        tokenizer_model=os.environ.get("EMBED_TOKENIZER_MODEL") or EMBED_MODEL_ID,
        dimension=model.get_sentence_embedding_dimension(),
        max_seq_length=max_length,
        normalized=True,
        protocol="research-embedding-v1",
        max_batch_size=EMBED_MAX_BATCH,
    )
    _models["embedder"] = model
    return model


def load_reranker_model() -> Any:
    if "reranker" not in _models:
        from sentence_transformers import CrossEncoder

        _models["reranker"] = CrossEncoder(RERANK_MODEL_ID, device=DEV_RERANK, max_length=512)
    return _models["reranker"]


InputText = Annotated[str, Field(min_length=1, max_length=16000)]
EmbeddingBatch = Annotated[list[InputText], Field(min_length=1, max_length=EMBED_MAX_BATCH)]


class EmbedRequest(BaseModel):
    texts: EmbeddingBatch | None = None
    input: InputText | EmbeddingBatch | None = None
    input_type: Literal["query", "passage"] | None = None
    is_query: bool = False
    model: str | None = None

    @model_validator(mode="after")
    def one_source(self) -> EmbedRequest:
        if (self.texts is None) == (self.input is None):
            raise ValueError("Provide exactly one of texts or input.")
        return self


class RerankRequest(BaseModel):
    query: InputText
    documents: Annotated[list[InputText], Field(max_length=64)]
    top_k: Annotated[int, Field(ge=1, le=64)] | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: InputText


class ChatCompletionRequest(BaseModel):
    model: str = "research-assist-local"
    messages: Annotated[list[ChatMessage], Field(min_length=1, max_length=64)]
    temperature: Annotated[float, Field(ge=0, le=2)] = 0.2
    max_tokens: Annotated[int, Field(ge=1, le=2048)] = 2048


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "online"}


@app.get("/api/admin/vram")
def vram(_auth: Authenticated) -> dict[str, Any]:
    return get_vram_stats()


@app.post("/api/admin/vram/clear")
def clear_vram(_auth: Authenticated) -> dict[str, Any]:
    if not CUDA_AVAILABLE or not torch.cuda.is_initialized():
        return {"status": "no_cuda_cache"}
    with idle_devices():
        if time.monotonic() < _quiesce_until:
            raise HTTPException(503, "GPU Worker đang dừng để bảo trì.")
        before = get_vram_stats()
        torch.cuda.empty_cache()
        return {"status": "cleared", "before": before, "after": get_vram_stats()}


@contextmanager
def idle_devices() -> Iterator[None]:
    with ExitStack() as stack:
        for lock in _locks.values():
            if not lock.acquire(blocking=False):
                raise HTTPException(409, "GPU Worker đang xử lý yêu cầu.")
            stack.callback(lock.release)
        yield


@app.post("/api/admin/worker/quiesce")
def quiesce_worker(_auth: Authenticated) -> dict[str, str]:
    global _quiesce_until
    with idle_devices():
        # Let the worker resume if the supervisor cannot finish stopping it.
        _quiesce_until = time.monotonic() + 20
    return {"status": "idle"}


@app.get("/v1/info")
def embedding_info(_auth: Authenticated) -> dict[str, Any]:
    with inference(DEV_EMBED):
        load_embedding_model()
    return dict(_embedding_info)


@app.post("/api/embed")
@app.post("/v1/embeddings")
def embed_endpoint(req: EmbedRequest, _auth: Authenticated) -> dict[str, Any]:
    inputs = req.texts if req.texts is not None else req.input
    texts = [inputs] if isinstance(inputs, str) else (inputs or [])
    if any(not text.strip() for text in texts):
        raise HTTPException(422, "Văn bản embedding cần có nội dung.")
    query = req.input_type == "query" if req.input_type is not None else req.is_query
    prefix = "query: " if query else "passage: "
    with inference(DEV_EMBED):
        model = load_embedding_model()
        model_id = _embedding_info["model"]
        if req.model and req.model not in {model_id, EMBED_MODEL_ID}:
            raise HTTPException(409, "Model embedding đã đổi. Cập nhật ID và lập lại chỉ mục.")
        vectors = model.encode(
            [prefix + text for text in texts],
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()
    return {
        "object": "list",
        "model": model_id if req.model != EMBED_MODEL_ID else EMBED_MODEL_ID,
        "embeddings": vectors,
        "data": [
            {"object": "embedding", "index": i, "embedding": vec} for i, vec in enumerate(vectors)
        ],
    }


@app.post("/api/rerank")
def rerank_endpoint(req: RerankRequest, _auth: Authenticated) -> dict[str, Any]:
    documents = req.documents
    if not documents:
        return {"model": RERANK_MODEL_ID, "results": []}
    with inference(DEV_RERANK):
        reranker = load_reranker_model()
        scores = reranker.predict(
            [[req.query, doc] for doc in documents], batch_size=8, show_progress_bar=False
        )
    ranked = sorted(
        (
            {"index": i, "document": doc, "relevance_score": float(score)}
            for i, (doc, score) in enumerate(zip(documents, scores, strict=True))
        ),
        key=lambda row: row["relevance_score"],
        reverse=True,
    )
    return {"model": RERANK_MODEL_ID, "results": ranked[: req.top_k]}


@app.get("/v1/models")
def list_models(_auth: Authenticated) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": "research-assist-local", "object": "model"}] if "llm" in _models else [],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, _auth: Authenticated) -> dict[str, Any]:
    llm = _models.get("llm")
    tokenizer = _tokenizers.get("llm")
    if llm is None or tokenizer is None:
        raise HTTPException(503, "GPU Worker chưa nạp LLM. Chọn worker LLM hoặc API đã cấu hình.")
    prompt_text = "\n".join(f"{message.role}: {message.content}" for message in req.messages)
    with inference(DEV_LLM):
        inputs = tokenizer(prompt_text, return_tensors="pt").to(DEV_LLM)
        options: dict[str, Any] = {
            "max_new_tokens": req.max_tokens,
            "do_sample": req.temperature > 0,
        }
        if req.temperature > 0:
            options["temperature"] = req.temperature
        outputs = llm.generate(**inputs, **options)
        generated = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=LOOPBACK, port=int(os.environ.get("WORKER_PORT", "8002")))
