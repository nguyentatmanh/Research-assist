"""Per-role model routing, exact validated cache, shared provider pacing and audit."""

import asyncio
import hashlib
import json
import time
from collections import deque
from collections.abc import Callable
from typing import Literal

import httpx
from pydantic import ValidationError

from research_assist.config import AgentRole, AgentRoute, Settings
from research_assist.domain.models import AgentUsage
from research_assist.domain.ports import LanguageModel, T
from research_assist.infrastructure.llm import DemoLanguageModel, HTTPLanguageModel, current_usage
from research_assist.infrastructure.repository import Repository

TASK_ROLES: dict[str, AgentRole] = {
    "retrieval_plan": "retrieval",
    "read_evidence": "reader",
    "synthesize_claims": "summarizer",
    "check_claims": "fact_checker",
    "write_report": "writer",
}


class RequestGate:
    """Conservative sliding-minute pacing shared by all routes of one provider."""

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self.timestamps: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self.lock:
                current = time.monotonic()
                while self.timestamps and current - self.timestamps[0] >= 60:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.rpm:
                    self.timestamps.append(current)
                    return
                wait = 60 - (current - self.timestamps[0])
            await asyncio.sleep(max(0.001, wait))


class ModelRouter:
    def __init__(
        self, settings: Settings, client: httpx.AsyncClient, repository: Repository
    ) -> None:
        self.settings, self.repository = settings, repository
        self.name: str = settings.llm_provider
        self.model: str = settings.resolved_model
        self.models: dict[AgentRole, LanguageModel] = {}
        self.routes: dict[AgentRole, Settings] = {}
        gates: dict[str, RequestGate] = {}
        for role in TASK_ROLES.values():
            route = settings.for_agent(role)
            self.routes[role] = route
            gate = gates.setdefault(
                route.llm_provider, RequestGate(settings.llm_requests_per_minute)
            )
            self.models[role] = (
                DemoLanguageModel()
                if route.llm_provider == "demo"
                else HTTPLanguageModel(route, client, gate)
            )
        providers = {m.name for m in self.models.values()}
        self.name = next(iter(providers)) if len(providers) == 1 else "mixed"
        if len({m.model for m in self.models.values()}) > 1:
            self.model = "per-agent routes"

    def public_routes(self) -> list[dict[str, object]]:
        return [
            {
                "agent": role,
                "provider": model.name,
                "model": model.model,
                "execution": self.execution(role),
                "max_output_tokens": self.routes[role].llm_max_output_tokens,
                "thinking_level": self.routes[role].gemini_thinking_level,
            }
            for role, model in self.models.items()
        ]

    def execution(self, role: AgentRole) -> Literal["local", "api", "demo"]:
        usage = current_usage.get()
        profile = usage.profile if usage and usage.profile else self.settings.workflow_profile
        if profile == "balanced" and role in {"retrieval", "writer"}:
            return "local"
        return "demo" if self.models[role].name == "demo" else "api"

    async def generate(
        self,
        *,
        task: str,
        instruction: str,
        payload: dict[str, object],
        schema: type[T],
        fallback: Callable[[], T],
    ) -> T:
        role = TASK_ROLES[task]
        model, route = self.models[role], self.routes[role]
        mode = self.execution(role)
        usage = current_usage.get()
        before = (usage.calls, usage.input_tokens, usage.output_tokens) if usage else (0, 0, 0)
        begin, hit = time.perf_counter(), False
        missing_before = usage.missing_usage if usage else 0
        try:
            if mode == "local":
                return schema.model_validate(fallback().model_dump())
            if mode == "demo":
                return await model.generate(
                    task=task,
                    instruction=instruction,
                    payload=payload,
                    schema=schema,
                    fallback=fallback,
                )
            cache_key = hashlib.sha256(
                json.dumps(
                    {
                        "version": 2,
                        "task": task,
                        "provider": model.name,
                        "model": model.model,
                        "endpoint": [
                            route.openai_base_url,
                            route.ollama_base_url,
                            route.deepseek_base_url,
                        ],
                        "instruction": instruction,
                        "payload": payload,
                        "schema": schema.model_json_schema(),
                        "output_tokens": route.llm_max_output_tokens,
                        "structured": route.structured_output,
                        "thinking": [
                            route.gemini_thinking_level,
                            route.gemini_thinking_budget,
                            route.deepseek_thinking,
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            cache_on = (
                self.settings.cache_enabled
                and self.settings.cache_ttl_seconds > 0
                and (usage is None or usage.cache_allowed)
            )
            if cache_on:
                cached = await asyncio.to_thread(self.repository.cache_get, cache_key)
                if cached:
                    try:
                        value = schema.model_validate_json(cached)
                        hit = True
                        if usage:
                            usage.cache_hits += 1
                        return value
                    except ValidationError:
                        pass
            value = await model.generate(
                task=task,
                instruction=instruction,
                payload=payload,
                schema=schema,
                fallback=fallback,
            )
            if cache_on:
                await asyncio.to_thread(
                    self.repository.cache_put,
                    cache_key,
                    value.model_dump_json(),
                    self.settings.cache_ttl_seconds,
                    self.settings.cache_max_entries,
                )
            return value
        finally:
            if usage:
                entry = usage.agents.setdefault(
                    role,
                    AgentUsage(
                        agent=role,
                        provider=model.name if mode != "local" else "local",
                        model=model.model if mode != "local" else "deterministic",
                        execution=mode,
                    ),
                )
                calls, inputs, outputs = (
                    usage.calls - before[0],
                    usage.input_tokens - before[1],
                    usage.output_tokens - before[2],
                )
                entry.calls += calls
                entry.input_tokens += inputs
                entry.output_tokens += outputs
                entry.cache_hits += int(hit)
                entry.elapsed_ms += round((time.perf_counter() - begin) * 1000, 2)
                prices = self.settings.agent_routes.get(role, AgentRoute())
                if mode != "api" or (hit and entry.calls == 0):
                    entry.estimated_cost_usd = 0.0
                elif (
                    prices.input_usd_per_million is not None
                    and prices.output_usd_per_million is not None
                    and usage.missing_usage == missing_before
                ):
                    entry.estimated_cost_usd = (entry.estimated_cost_usd or 0.0) + (
                        inputs * prices.input_usd_per_million
                        + outputs * prices.output_usd_per_million
                    ) / 1_000_000
                else:
                    entry.estimated_cost_usd = None
