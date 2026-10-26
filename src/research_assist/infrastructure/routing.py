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
from research_assist.domain.errors import ProviderUnavailable
from research_assist.domain.metadata import EVIDENCE_POLICY
from research_assist.domain.models import AgentUsage
from research_assist.domain.ports import LanguageModel, T
from research_assist.infrastructure.llm import (
    DemoLanguageModel,
    HTTPLanguageModel,
    Usage,
    current_call_usage,
    current_usage,
    provider_event,
)
from research_assist.infrastructure.repository import Repository

TASK_ROLES: dict[str, AgentRole] = {
    "retrieval_brief": "retrieval",
    "reader_brief": "reader",
    "summarizer_brief": "summarizer",
    "fact_checker_brief": "fact_checker",
    "writer_brief": "writer",
    "retrieval_step": "retrieval",
    "reader_step": "reader",
    "summarizer_step": "summarizer",
    "fact_checker_step": "fact_checker",
    "writer_step": "writer",
    "retrieval_action": "retrieval",
    "question_plan": "retrieval",
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
        self.fallback_models: dict[AgentRole, LanguageModel] = {}
        self.routes: dict[AgentRole, Settings] = {}
        gates: dict[str, RequestGate] = {}
        for role in dict.fromkeys(TASK_ROLES.values()):
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
            if route.gemini_fallback_model:
                secondary = route.model_copy(update={"llm_model": route.gemini_fallback_model})
                self.fallback_models[role] = HTTPLanguageModel(secondary, client, gate)
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
                "fallback_model": self.routes[role].gemini_fallback_model or None,
                "execution": self.execution(role),
                "planner_execution": ("demo" if model.name == "demo" else "api")
                if role == "retrieval"
                else None,
                "max_output_tokens": self.routes[role].llm_max_output_tokens,
                "thinking_level": self.routes[role].gemini_thinking_level,
            }
            for role, model in self.models.items()
        ]

    def execution(self, role: AgentRole) -> Literal["local", "api", "demo"]:
        usage = current_usage.get()
        profile = usage.profile if usage and usage.profile else self.settings.workflow_profile
        orchestration = (
            usage.orchestration_mode
            if usage and usage.orchestration_mode
            else self.settings.orchestration_mode
        )
        if orchestration == "autonomous":
            return "demo" if self.models[role].name == "demo" else "api"
        if role == "writer" and orchestration == "adaptive":
            return "local"
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
        if task in {"question_plan", "retrieval_action"} or task.endswith(("_step", "_brief")):
            mode = "demo" if model.name == "demo" else "api"
        usage = current_usage.get()
        call_usage = Usage()
        call_token = current_call_usage.set(call_usage)
        begin, hit = time.perf_counter(), False
        used_models = [model.model]
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
                        "policy": EVIDENCE_POLICY,
                        "task": task,
                        "provider": model.name,
                        "model": model.model,
                        "fallback_model": route.gemini_fallback_model,
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
                        "wire_schema_policy": "gemini-compact-bounds-v1",
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
            secondary_model = self.fallback_models.get(role)
            unavailable = (model.name, model.model)
            skip_primary = bool(
                secondary_model and usage and unavailable in usage.unavailable_models
            )
            primary_skipped = skip_primary
            if not skip_primary:
                try:
                    value = await model.generate(
                        task=task,
                        instruction=instruction,
                        payload=payload,
                        schema=schema,
                        fallback=fallback,
                    )
                except ProviderUnavailable:
                    if usage:
                        usage.unavailable_models.add(unavailable)
                    if secondary_model is None:
                        raise
                    skip_primary = True
            if skip_primary and secondary_model is not None:
                provider_event(
                    task,
                    "model_fallback",
                    provider=model.name,
                    primary_model=model.model,
                    fallback_model=secondary_model.model,
                    primary_skipped=primary_skipped,
                )
                # State is per run; no shared mutation of the configured primary model.
                if primary_skipped:
                    used_models = []
                used_models.append(secondary_model.model)
                value = await secondary_model.generate(
                    task=task,
                    instruction=instruction,
                    payload=payload,
                    schema=schema,
                    fallback=fallback,
                )
            # Do not label fallback content as a cached answer from the primary model.
            if cache_on and not skip_primary:
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
                if mode == "api" and not hit:
                    used_models = list(
                        dict.fromkeys(
                            str(event["model"])
                            for event in call_usage.provider_events
                            if event["kind"] in {"http_attempt", "network_error"}
                            and "model" in event
                        )
                    )
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
                    call_usage.calls,
                    call_usage.input_tokens,
                    call_usage.output_tokens,
                )
                entry.calls += calls
                entry.input_tokens += inputs
                entry.output_tokens += outputs
                entry.cache_hits += int(hit)
                for model_name in used_models:
                    if model_name not in entry.models_used:
                        entry.models_used.append(model_name)
                if mode == "api" and entry.models_used:
                    entry.model = " → ".join(entry.models_used)
                entry.elapsed_ms += round((time.perf_counter() - begin) * 1000, 2)
                prices = self.settings.agent_routes.get(role, AgentRoute())
                if mode != "api" or (hit and entry.calls == 0):
                    entry.estimated_cost_usd = 0.0
                elif (
                    prices.input_usd_per_million is not None
                    and prices.output_usd_per_million is not None
                    and call_usage.missing_usage == 0
                    and entry.models_used == [self.models[role].model]
                ):
                    entry.estimated_cost_usd = (entry.estimated_cost_usd or 0.0) + (
                        inputs * prices.input_usd_per_million
                        + outputs * prices.output_usd_per_million
                    ) / 1_000_000
                else:
                    entry.estimated_cost_usd = None
            current_call_usage.reset(call_token)
