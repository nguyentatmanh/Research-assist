"""Bind structural citations to the exact source and checked claim snapshot."""

from research_assist.agents.policies.verification import validate_checks
from research_assist.application.research_state import ResearchState
from research_assist.application.task_execution import fingerprint
from research_assist.domain.models import CitationAudit, ClaimCheck, RetrievedChunk
from research_assist.domain.provenance import evidence_source_hash, intersect_regions


def audit_run_sources(
    state: ResearchState,
    document_ids: list[str],
    versions: dict[str, str],
) -> None:
    """Recheck every published quote against the source window used by the run.

    The parallel coordinator keeps all task-local windows and audits those itself.
    Other modes keep one retrieved window per chunk in the report state.
    """
    ids = [check.claim_id for check in state.checks if check.verdict == "supported"]
    if not ids:
        return
    allowed = set(document_ids)
    sources = {
        chunk.id: [chunk]
        for chunk in state.retrieved.values()
        if not allowed or chunk.document_id in allowed
    }
    audit = audit_citations(state, sources, versions, verification_snapshot(state), ids)
    audit.rejected_claim_ids = list(
        dict.fromkeys([*state.citation_audit.rejected_claim_ids, *audit.rejected_claim_ids])
    )
    audit.reasons = list(dict.fromkeys([*state.citation_audit.reasons, *audit.reasons]))
    state.citation_audit = audit
    remove_rejected_claims(state, audit)


def verification_snapshot(state: ResearchState) -> str:
    return fingerprint(
        [
            [c.model_dump() for c in state.claims],
            [e.model_dump() for e in state.evidence],
            [c.model_dump() for c in state.checks],
            [u.model_dump() for u in state.evidence_usages],
        ]
    )


def audit_citations(
    state: ResearchState,
    sources: dict[str, list[RetrievedChunk]],
    versions: dict[str, str],
    approved_snapshot: str,
    reported_ids: list[str],
) -> CitationAudit:
    evidence = {e.id: e for e in state.evidence}
    checks = {c.claim_id: c for c in validate_checks(state.claims, state.evidence, state.checks)}
    claims = {c.id: c for c in state.claims}
    fresh = verification_snapshot(state) == approved_snapshot
    accepted, rejected, reasons = [], [], []
    for cid in reported_ids:
        check = checks.get(cid)
        reason = ""
        if not fresh or cid not in claims or not check or check.verdict != "supported":
            reason = "Claim does not have a current approved verification snapshot."
        elif not check.evidence_ids:
            reason = "Claim has no checked citation."
        else:
            for eid in check.evidence_ids:
                ev = evidence.get(eid)
                if ev is None or ev.document_version != versions.get(ev.document_id, ""):
                    reason = "Evidence source version is invalid."
                    break
                if ev.source_hash and ev.source_hash != evidence_source_hash(ev):
                    reason = "Evidence source hash is invalid."
                    break
                windows = sources.get(ev.chunk_id, [])
                if not any(
                    c.document_id == ev.document_id
                    and c.page == ev.page
                    and c.filename == ev.filename
                    and c.section == ev.section
                    and c.start <= ev.start < ev.end <= c.end
                    and ev.quote == c.text[ev.start - c.start : ev.end - c.start]
                    and ev.regions == intersect_regions(c.regions, ev.start, ev.end)
                    for c in windows
                ):
                    reason = "Evidence quote or source locator is invalid."
                    break
        if reason:
            rejected.append(cid)
            reasons.append(f"{cid}: {reason}")
        else:
            accepted.append(cid)
    return CitationAudit(
        # "Passed" describes only the admitted output. If every candidate fails,
        # report the failure explicitly; mixed audits retain the rejection reasons.
        status="passed" if accepted else "failed" if rejected else "not_applicable",
        snapshot=approved_snapshot,
        accepted_claim_ids=accepted,
        rejected_claim_ids=rejected,
        reasons=reasons,
    )


def published_citation_audit(audit: CitationAudit, claim_ids: list[str]) -> CitationAudit:
    published = set(claim_ids)
    accepted = [claim_id for claim_id in audit.accepted_claim_ids if claim_id in published]
    status = (
        "passed" if accepted else "failed" if audit.rejected_claim_ids else "not_applicable"
    )
    return audit.model_copy(update={"status": status, "accepted_claim_ids": accepted})


def remove_rejected_claims(state: ResearchState, audit: CitationAudit) -> None:
    rejected = set(audit.rejected_claim_ids)
    state.checks = [
        ClaimCheck(
            claim_id=c.claim_id,
            verdict="insufficient",
            evidence_ids=[],
            explanation="Citation audit rejected the source or verification snapshot.",
        )
        if c.claim_id in rejected
        else c
        for c in state.checks
    ]
    state.warnings.extend(audit.reasons)
