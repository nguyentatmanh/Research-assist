"""Private persisted GPU endpoint settings and bounded connection checks."""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from research_assist.config import Settings
from research_assist.domain.errors import ResearchError
from research_assist.infrastructure.embeddings import validate_vectors

OVERRIDE_FIELDS = frozenset(
    {
        "embedding_provider",
        "embedding_base_url",
        "embedding_model",
        "embedding_api_key",
        "embedding_batch_size",
        "embedding_max_seq_length",
        "chunk_strategy",
        "chunk_tokens",
        "chunk_token_overlap",
        "chunk_tokenizer_model",
        "chunk_tokenizer_revision",
    }
)


class RuntimeSettings(Settings):
    """Validate a complete runtime snapshot without re-reading .env or process env."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


class RemoteEmbeddingInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    base_url: str = Field(min_length=1, max_length=2048)
    api_key: SecretStr = SecretStr("")
    model_id: str = Field(default="", max_length=160)
    model_family: str = Field(default="auto", pattern="^(auto|e5|qwen3)$")
    tokenizer_model: str = Field(
        default="intfloat/multilingual-e5-base", min_length=1, max_length=300
    )
    batch_size: int = Field(default=16, ge=1, le=32)


def normalize_endpoint(value: str) -> str:
    value = value.strip().rstrip("/")
    try:
        parsed = urlsplit(value)
        _ = parsed.port
        httpx.URL(value)
    except (ValueError, httpx.InvalidURL):
        raise ResearchError("URL GPU endpoint không hợp lệ.") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ResearchError("URL endpoint không hợp lệ; dán URL gốc hoặc URL kết thúc bằng /v1.")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ResearchError("Endpoint bên ngoài máy cần dùng HTTPS.")
    if parsed.path not in {"", "/v1"}:
        raise ResearchError("Dán URL gốc hoặc URL /v1, không dán đường dẫn /embeddings hay /info.")
    return value if value.endswith("/v1") else value + "/v1"


def settings_path(settings: Settings) -> Path:
    return settings.data_dir / "remote-embedding.json"


def load_saved_settings(settings: Settings) -> Settings:
    path = settings_path(settings)
    if not path.exists():
        return settings
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict) or set(values) - OVERRIDE_FIELDS:
            raise ValueError("Invalid configuration fields")
        if values.get("embedding_provider") != "remote":
            raise ValueError("Invalid provider")
        return RuntimeSettings(**{**settings.model_dump(), **values})
    except (ValueError, OSError):
        raise ResearchError("Không đọc được cấu hình GPU đã lưu trong thư mục data.") from None


def persist_settings(settings: Settings) -> None:
    path = settings_path(settings)
    values: dict[str, Any] = {name: getattr(settings, name) for name in OVERRIDE_FIELDS}
    values["embedding_api_key"] = settings.embedding_api_key.get_secret_value()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement; private file, never included in API responses or browser storage.
    descriptor, temporary = tempfile.mkstemp(prefix=".gpu-settings-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(values, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


async def probe_endpoint(
    body: RemoteEmbeddingInput, current: Settings, client: httpx.AsyncClient
) -> tuple[Settings, dict[str, object]]:
    url = normalize_endpoint(body.base_url)
    key = body.api_key.get_secret_value().strip()
    if not key:
        if current.embedding_provider != "remote" or url != normalize_endpoint(
            current.embedding_base_url
        ):
            raise ResearchError("Nhập key truy cập GPU; cần nhập lại key khi đổi URL.")
        key = current.embedding_api_key.get_secret_value()
    if not 32 <= len(key) <= 4096 or "\n" in key or "\r" in key:
        raise ResearchError("Key truy cập GPU cần từ 32 đến 4096 ký tự trên một dòng.")
    begin = time.perf_counter()
    headers = {"Authorization": "Bearer " + key}
    try:
        response = await client.get(url + "/info", headers=headers, timeout=15)
        response.raise_for_status()
        info = response.json()
        if not isinstance(info, dict):
            raise ResearchError("Endpoint không trả thông tin checkpoint hợp lệ.")
        model = info.get("model", "")
        length, dimension = info.get("max_seq_length"), info.get("dimension")
        family = info.get("family")
        if (
            info.get("protocol") != "research-embedding-v1"
            or family not in {"e5", "qwen3"}
            or info.get("normalized") is not True
            or not isinstance(model, str)
            or not model.startswith("research-" + family + "-")
            or len(model) > 160
            or type(length) is not int
            or not 80 <= length <= 8192
            or type(dimension) is not int
            or not 1 <= dimension <= 16384
        ):
            raise ResearchError(
                "Endpoint không đúng giao thức embedding của notebook Research-assist."
            )
        if body.model_family != "auto" and body.model_family != family:
            raise ResearchError("Họ model đã chọn không khớp checkpoint trên GPU.")
        server_tokenizer = info.get("tokenizer_model")
        if (
            family == "qwen3"
            and isinstance(server_tokenizer, str)
            and server_tokenizer.startswith("Qwen/Qwen3-Embedding-")
            and body.tokenizer_model.startswith("Qwen/Qwen3-Embedding-")
            and body.tokenizer_model != server_tokenizer
        ):
            raise ResearchError("Tokenizer Qwen đã chọn không khớp model trên GPU.")
        if body.model_id and body.model_id != model:
            raise ResearchError("Model ID không khớp checkpoint đang chạy trên GPU.")
        candidate = RuntimeSettings(
            **{
                **current.model_dump(),
                "embedding_provider": "remote",
                "embedding_base_url": url,
                "embedding_model": model,
                "embedding_api_key": key,
                "embedding_batch_size": body.batch_size,
                "embedding_max_seq_length": length,
                "chunk_strategy": "tokens",
                "chunk_tokens": min(384, length - 16),
                "chunk_token_overlap": 48,
                "chunk_tokenizer_model": body.tokenizer_model,
                "chunk_tokenizer_revision": "",
            }
        )
        response = await client.post(
            url + "/embeddings",
            headers=headers,
            timeout=20,
            json={"model": model, "input": ["Kiểm tra kết nối GPU."], "input_type": "query"},
        )
        response.raise_for_status()
        payload = response.json()
        if (
            not isinstance(payload, dict)
            or payload.get("model") != model
            or len(payload.get("data", [])) != 1
        ):
            raise ResearchError("GPU trả phản hồi embedding không hợp lệ.")
        row = payload["data"][0]
        if not isinstance(row, dict) or row.get("index") != 0:
            raise ResearchError("GPU trả chỉ số embedding không hợp lệ.")
        vectors = validate_vectors([row["embedding"]], 1)
        if len(vectors[0]) != dimension:
            raise ResearchError("Số chiều embedding không khớp thông tin checkpoint.")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {401, 403}:
            raise ResearchError("Key truy cập GPU không đúng hoặc không được phép.") from None
        raise ResearchError("GPU endpoint chưa sẵn sàng. Kiểm tra server và URL tunnel.") from None
    except (httpx.HTTPError, ValueError, KeyError, TypeError, ValidationError):
        raise ResearchError(
            "Không kiểm tra được GPU endpoint. Kiểm tra URL, notebook và kết nối."
        ) from None
    return candidate, {
        "connected": True,
        "model_id": model,
        "model_family": family,
        "tokenizer_model": server_tokenizer if isinstance(server_tokenizer, str) else "",
        "dimension": dimension,
        "max_seq_length": length,
        "base_url": url,
        "latency_ms": round((time.perf_counter() - begin) * 1000),
    }
