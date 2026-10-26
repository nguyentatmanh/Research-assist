"""Per-run working state. Only complete verification passes advance the report snapshot."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from research_assist.domain.models import (
    AgentSession,
    AgentTask,
    Claim,
    ClaimCheck,
    CoverageCheck,
    Evidence,
    EvidenceGap,
    FollowUpRequest,
    PeerMessage,
    RetrievedChunk,
    RuntimeTask,
    WriterOutput,
)


@dataclass
class ResearchState:
    retrieved: dict[str, RetrievedChunk] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    checks: list[ClaimCheck] = field(default_factory=list)
    coverage: list[CoverageCheck] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    gap_details: list[EvidenceGap] = field(default_factory=list)
    requests: list[FollowUpRequest] = field(default_factory=list)
    tasks: dict[str, AgentTask] = field(default_factory=dict)
    history: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rounds: int = 1
    stop_reason: str = "round_limit"
    task_graph: dict[str, RuntimeTask] = field(default_factory=dict)
    sessions: dict[str, AgentSession] = field(default_factory=dict)
    report_layout: WriterOutput | None = None
    agent_plans: dict[str, str] = field(default_factory=dict)
    peer_messages: list[PeerMessage] = field(default_factory=list)
    checkpoint: Callable[[], Awaitable[None]] | None = None

    async def save_checkpoint(self) -> None:
        if self.checkpoint is not None:
            await self.checkpoint()
