"""Named execution targets: private persistence and idle-only replacement."""

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from research_assist.application.embedding_settings import RuntimeSettings
from research_assist.config import ExecutionTarget, Settings
from research_assist.domain.errors import ResearchError

EditableTarget = Literal["api", "cpu", "gpu"]


class TargetInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    provider: Literal["gemini", "openai", "ollama", "deepseek"]
    model: str = Field(min_length=1, max_length=200)
    base_url: str = Field(default="", max_length=2048)
    api_key: SecretStr = SecretStr("")


def target_settings_path(settings: Settings) -> Path:
    return settings.data_dir / "execution-targets.json"


def load_targets(settings: Settings) -> Settings:
    path = target_settings_path(settings)
    if not path.exists():
        return settings
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict) or set(values) - {"api", "cpu", "gpu"}:
            raise ValueError("Invalid target names")
        overrides = {
            name: value
            for name, value in values.items()
            if name not in settings.managed_execution_targets
        }
        return RuntimeSettings(
            **{
                **settings.model_dump(),
                "execution_targets": {**settings.execution_targets, **overrides},
            }
        )
    except (ValueError, OSError):
        raise ResearchError("Không đọc được cấu hình đích model trong thư mục data.") from None


def updated_target(current: Settings, name: EditableTarget, body: TargetInput) -> Settings:
    if name in current.managed_execution_targets:
        raise ResearchError("Đích này do boot quản lý; hãy sửa cấu hình boot và khởi động lại.")
    old = current.execution_targets.get(name)
    key = body.api_key.get_secret_value().strip()
    base_url = body.base_url.rstrip("/")
    if not key and old and old.provider == body.provider and old.base_url.rstrip("/") == base_url:
        key = old.api_key.get_secret_value()
    if len(key) > 4096 or "\n" in key or "\r" in key:
        raise ResearchError("Token không hợp lệ; dùng một dòng tối đa 4096 ký tự.")
    try:
        target = ExecutionTarget(
            provider=body.provider, model=body.model, base_url=base_url, api_key=SecretStr(key)
        )
        return RuntimeSettings(
            **{
                **current.model_dump(),
                "execution_targets": {**current.execution_targets, name: target},
            }
        )
    except ValidationError:
        raise ResearchError(
            "Kiểm tra provider, model, URL HTTPS hoặc localhost và token. "
            "Khi đổi endpoint/provider cần nhập token mới."
        ) from None


def persist_targets(settings: Settings) -> None:
    path = target_settings_path(settings)
    values = {}
    for name, target in settings.execution_targets.items():
        if name not in settings.managed_execution_targets:
            values[name] = {**target.model_dump(), "api_key": target.api_key.get_secret_value()}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".execution-targets-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(values, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def public_target_config(settings: Settings) -> list[dict[str, object]]:
    return [
        {
            "id": name,
            "provider": target.provider,
            "model": target.model,
            "base_url": target.base_url,
            "key_configured": bool(target.api_key.get_secret_value()),
            "managed": name in settings.managed_execution_targets,
        }
        for name, target in settings.execution_targets.items()
    ]
