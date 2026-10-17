import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import httpx

from research_assist.config import Settings
from research_assist.domain.errors import ProviderError
from research_assist.infrastructure.local_runtime import INFERENCE_LOCK


def validate_vectors(vectors: object, count: int) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ProviderError("Embedding trả về số vector không đúng.")
    result: list[list[float]] = []
    dimension: int | None = None
    for vector in vectors:
        if not isinstance(vector, list) or not vector or len(vector) > 16384:
            raise ProviderError("Embedding trả về vector không hợp lệ.")
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in vector
        ):
            raise ProviderError("Embedding có chiều hoặc giá trị không hợp lệ.")
        # hypot scales internally, avoiding overflow for large but finite API values.
        norm = math.hypot(*(float(v) for v in vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ProviderError("Embedding trả về vector rỗng hoặc không hữu hạn.")
        result.append([float(v) / norm for v in vector])
    return result


class NoEmbeddings:
    fingerprint: str | None = None

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        return []


class LocalEmbeddings:
    def __init__(
        self,
        model: str,
        batch_size: int,
        *,
        family: str = "auto",
        device: str = "cpu",
        revision: str = "",
        max_seq_length: int = 512,
    ) -> None:
        manifest_path = Path(model) / "research_model.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        if family == "auto":
            family = manifest.get("model_family") or (
                "e5"
                if "e5" in model.lower()
                else "qwen3"
                if "qwen3" in model.lower()
                else "generic"
            )
        self.family, self.device, self.revision = family, device, revision
        self.max_seq_length = max_seq_length
        signature = checkpoint_signature(Path(model)) if Path(model).is_dir() else "remote"
        self.fingerprint: str | None = (
            f"sentence-transformers:{model}:{family}:{revision}:{signature}:{max_seq_length}:v2"
        )
        self.model_name, self.batch_size = model, batch_size
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._encode, texts, query)

    def _encode(self, texts: list[str], query: bool) -> list[list[float]]:
        with INFERENCE_LOCK:
            return self._encode_locked(texts, query)

    def _encode_locked(self, texts: list[str], query: bool) -> list[list[float]]:
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                options: dict[str, Any] = {"trust_remote_code": False}
                if self.device != "auto":
                    options["device"] = self.device
                if self.revision:
                    options["revision"] = self.revision
                self._model = SentenceTransformer(self.model_name, **options)
                self._model.max_seq_length = self.max_seq_length
            # E5 requires these prefixes even for non-English input.
            if self.family == "e5":
                texts = [("query: " if query else "passage: ") + t for t in texts]
            elif self.family == "qwen3" and query:
                texts = [
                    "Instruct: Retrieve passages that answer the research question.\nQuery: " + t
                    for t in texts
                ]
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            return validate_vectors(vectors, len(texts))
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                "Không tải/chạy được embedding cục bộ. Cài extra local-models và tải model trước."
            ) from exc


def checkpoint_signature(path: Path) -> str:
    """Detect replacing weights at the same local path; never reuse old vectors silently."""
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix in {".json", ".safetensors", ".bin"}:
            digest.update(str(file.relative_to(path)).encode())
            with file.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
    return digest.hexdigest()[:24]


class APIEmbeddings:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings, self.client = settings, client
        self.fingerprint: str | None = (
            f"openai:{settings.embedding_base_url}:{settings.embedding_model}:normalized:v1"
        )

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        all_vectors = []
        for start in range(0, len(texts), self.settings.embedding_batch_size):
            batch = texts[start : start + self.settings.embedding_batch_size]
            try:
                response = await self.client.post(
                    f"{self.settings.embedding_base_url}/embeddings",
                    headers={
                        "Authorization": "Bearer "
                        + self.settings.embedding_api_key.get_secret_value()
                    },
                    json={"model": self.settings.embedding_model, "input": batch},
                )
                response.raise_for_status()
                payload = response.json()["data"]
                rows = sorted(payload, key=lambda row: row["index"])
                if [row["index"] for row in rows] != list(range(len(batch))):
                    raise ProviderError("Embedding trả về chỉ số vector không đúng.")
                all_vectors.extend(validate_vectors([row["embedding"] for row in rows], len(batch)))
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(
                    "Embedding API không khả dụng hoặc phản hồi sai định dạng."
                ) from exc
        return validate_vectors(all_vectors, len(texts)) if texts else []
