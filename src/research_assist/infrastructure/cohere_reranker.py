"""Cohere v2 Rerank: batched documents, bounded pacing, exact cache, strict provenance."""

import asyncio
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
from pydantic import BaseModel, Field, ValidationError

from research_assist.config import Settings
from research_assist.domain.errors import ProviderError
from research_assist.domain.models import RetrievedChunk
from research_assist.infrastructure.llm import current_usage
from research_assist.infrastructure.repository import Repository
from research_assist.infrastructure.routing import RequestGate


class RankedItem(BaseModel):
    index: int = Field(strict=True, ge=0)
    relevance_score: float = Field(strict=True, ge=0, le=1, allow_inf_nan=False)


class RankedResponse(BaseModel):
    results: list[RankedItem]


class CohereReranker:
    provider = "cohere"

    def __init__(self, settings: Settings, client: httpx.AsyncClient, repository: Repository):
        self.settings, self.client, self.repository = settings, client, repository
        self.model_name = settings.resolved_reranker_model
        self.gate = RequestGate(settings.reranker_requests_per_minute)
        self._lock = asyncio.Lock()
        self._blocked_until = 0.0
        self.calls = self.cache_hits = self.failures = 0
        self.last_rank: dict[str, object] = {}

    def diagnostics(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model_name,
            "api_calls": self.calls,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "requests_per_minute": self.settings.reranker_requests_per_minute,
        }

    @staticmethod
    def _decode(payload: object, count: int) -> RankedResponse:
        response = RankedResponse.model_validate(payload)
        if len(response.results) != count or {r.index for r in response.results} != set(
            range(count)
        ):
            raise ValueError("Incomplete, duplicate or foreign indices")
        return response

    async def rank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        # Serializing cache misses also coalesces concurrent identical requests.
        async with self._lock:
            self.last_rank = {
                "provider": self.provider,
                "model": self.model_name,
                "cache_hit": False,
                "api_calls": 0,
            }
            try:
                return await self._rank(query, chunks)
            except ProviderError:
                self.failures += 1
                self.last_rank["failed"] = True
                raise

    async def _rank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [c.text for c in chunks],
            "top_n": len(chunks),
            "max_tokens_per_doc": self.settings.reranker_max_tokens_per_doc,
        }
        cache_key = (
            "cohere-rerank:"
            + hashlib.sha256(
                json.dumps(
                    [self.settings.cohere_base_url, payload], ensure_ascii=False, sort_keys=True
                ).encode()
            ).hexdigest()
        )
        usage = current_usage.get()
        cache_on = (
            self.settings.cache_enabled
            and self.settings.reranker_cache_ttl_seconds > 0
            and (usage is None or usage.cache_allowed)
        )
        response = None
        if cache_on:
            cached = await asyncio.to_thread(self.repository.cache_get, cache_key)
            if cached:
                try:
                    response = self._decode(json.loads(cached), len(chunks))
                    self.cache_hits += 1
                    self.last_rank["cache_hit"] = True
                except (ValidationError, ValueError, TypeError):
                    pass
        if response is None:
            cooldown = self._blocked_until - time.monotonic()
            if cooldown > 0:
                raise ProviderError("Cohere Rerank đang tạm ngưng gọi lại sau lỗi hạn mức.")
            await self.gate.acquire()
            try:
                self.calls += 1
                self.last_rank["api_calls"] = 1
                result = await self.client.post(
                    self.settings.cohere_base_url + "/rerank",
                    headers={
                        "Authorization": "Bearer " + self.settings.cohere_api_key.get_secret_value()
                    },
                    json=payload,
                    timeout=self.settings.reranker_timeout_seconds,
                )
            except httpx.HTTPError:
                raise ProviderError(
                    "Không kết nối được Cohere Rerank hoặc yêu cầu hết thời gian."
                ) from None
            if result.status_code == 429:
                delay = self._retry_delay(result.headers.get("Retry-After", "60"))
                self._blocked_until = time.monotonic() + delay
                raise ProviderError(
                    "Cohere Rerank vượt hạn mức tốc độ hoặc quota. Kiểm tra Billing & Usage."
                )
            if result.status_code in {401, 403, 498}:
                raise ProviderError("Cohere Rerank từ chối key hoặc quyền truy cập model.")
            if not result.is_success:
                raise ProviderError(f"Cohere Rerank không hoàn thành (HTTP {result.status_code}).")
            try:
                response = self._decode(result.json(), len(chunks))
            except (ValidationError, ValueError, TypeError):
                raise ProviderError(
                    "Cohere Rerank trả chỉ số hoặc điểm xếp hạng không hợp lệ."
                ) from None
            if cache_on:
                await asyncio.to_thread(
                    self.repository.cache_put,
                    cache_key,
                    response.model_dump_json(),
                    self.settings.reranker_cache_ttl_seconds,
                    self.settings.cache_max_entries,
                )
        scored = [
            chunks[item.index].model_copy(update={"rerank_score": item.relevance_score})
            for item in response.results
        ]
        return sorted(scored, key=lambda c: -(c.rerank_score or 0))

    @staticmethod
    def _retry_delay(value: str) -> float:
        try:
            seconds = float(value)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                seconds = 60
        # A cooldown fails fast; honor the server delay without blocking an entire run.
        return max(1.0, seconds) if math.isfinite(seconds) else 60.0
