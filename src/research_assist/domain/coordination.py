"""Validate inter-agent work requests without giving models authority over verdicts."""

from hashlib import sha256

from research_assist.domain.models import (
    AgentTask,
    Claim,
    CoverageCheck,
    Evidence,
    FollowUpRequest,
    Requirement,
    SupervisorDecision,
)

ASSIGNEES = {
    "retrieve": "retrieval",
    "read_context": "reader",
    "revise": "summarizer",
    "verify": "fact_checker",
}


def valid_requests(
    requests: list[FollowUpRequest],
    requirements: list[Requirement],
    claims: list[Claim],
    evidence: list[Evidence],
) -> list[FollowUpRequest]:
    rids, cids = {r.id for r in requirements}, {c.id for c in claims}
    chunks = {e.chunk_id for e in evidence}
    accepted = []
    for item in requests:
        if (
            not set(item.requirement_ids).issubset(rids)
            or not set(item.claim_ids).issubset(cids)
            or not set(item.chunk_ids).issubset(chunks)
            or (item.action == "read_context" and not item.chunk_ids)
            or (item.action == "revise" and not item.claim_ids)
        ):
            continue
        # A model cannot use a valid requirement ID to target somebody else's claim.
        selected = [c for c in claims if c.id in item.claim_ids]
        if any(
            c.requirement_ids and not set(c.requirement_ids).intersection(item.requirement_ids)
            for c in selected
        ):
            continue
        accepted.append(item)
    return accepted


def work_requests(
    requests: list[FollowUpRequest],
    coverage: list[CoverageCheck],
    requirements: list[Requirement],
) -> list[FollowUpRequest]:
    """A malformed/omitted work request must not turn missing coverage into success."""
    pending = {row.requirement_id for row in coverage if row.status != "covered"}
    result = [r for r in requests if pending.intersection(r.requirement_ids)]
    assigned = {rid for r in result for rid in r.requirement_ids}
    result.extend(
        FollowUpRequest(
            action="retrieve",
            requirement_ids=[r.id],
            need="Find evidence to answer this requested aspect: " + r.text,
        )
        for r in requirements
        if r.id in pending - assigned
    )
    return result[:12]


def make_task(request: FollowUpRequest) -> AgentTask:
    digest = sha256(request.model_dump_json().encode()).hexdigest()[:20]
    return AgentTask.model_validate(
        {
            **request.model_dump(),
            "id": "task_" + digest,
            "assigned_to": ASSIGNEES[request.action],
        }
    )


def selected_tasks(decision: SupervisorDecision, tasks: list[AgentTask]) -> list[AgentTask] | None:
    """None is an invalid dispatch; an empty list is a valid explicit finish."""
    if decision.action == "finish":
        return [] if not decision.task_ids else None
    if not decision.task_ids or len(set(decision.task_ids)) != len(decision.task_ids):
        return None
    lookup = {t.id: t for t in tasks if t.status in {"pending", "unresolved"}}
    if any(tid not in lookup for tid in decision.task_ids):
        return None
    chosen = [lookup[tid] for tid in decision.task_ids]
    if any(t.action != decision.action for t in chosen):
        return None
    return chosen
