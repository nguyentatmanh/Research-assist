"""The shared language between agents. Identifiers are assigned by the application."""

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, str_strip_whitespace=True, allow_inf_nan=False
    )


class Document(Contract):
    id: str
    filename: str
    title: str
    kind: str
    sha256: str
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    created_at: str
    warnings: list[str] = Field(default_factory=list)
    index_version: str = "legacy"


class SourceRegion(Contract):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    bbox: list[float] = Field(min_length=4, max_length=4)


class SourceSection(Contract):
    text: str
    page: int | None = None
    label: str = "Văn bản"
    offset: int = 0
    extraction: Literal["text", "ocr"] = "text"
    warnings: list[str] = Field(default_factory=list)
    regions: list[SourceRegion] = Field(default_factory=list)


class Chunk(Contract):
    id: str
    document_id: str
    document_title: str
    filename: str
    text: str
    page: int | None
    section: str
    ordinal: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    regions: list[SourceRegion] = Field(default_factory=list)


class IndexedChunk(Chunk):
    vector: list[float] | None = None
    vector_model: str | None = None


class RetrievedChunk(Chunk):
    score: float = Field(ge=0)
    lexical_score: float = 0.0
    semantic_score: float | None = None
    rerank_score: float | None = None


class ResearchRequest(Contract):
    question: str = Field(min_length=8, max_length=2000)
    # A frozen all-document scope can contain the entire corpus, including over 50 files.
    # Membership is validated against the repository when the run is submitted.
    document_ids: list[str] = Field(default_factory=list)
    scope_mode: Literal["all", "selected"] = "all"
    top_k: int = Field(default=8, ge=2, le=12)
    language: Literal["vi", "en"] = "vi"
    style: Literal["overview", "comparison", "literature_review"] = "overview"
    retrieval_mode: Literal["lexical", "hybrid"] = "hybrid"
    max_rounds: int = Field(default=2, ge=1, le=2)
    profile: Literal["balanced", "academic"] | None = None
    use_cache: bool = True
    orchestration_mode: Literal["fixed", "adaptive", "autonomous"] | None = None
    requirements: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def selected_scope(self) -> "ResearchRequest":
        if self.scope_mode == "selected" and not self.document_ids:
            raise ValueError("Selected scope requires at least one document")
        return self

    @field_validator("requirements")
    @classmethod
    def requirement_texts(cls, values: list[str]) -> list[str]:
        if any(not v.strip() or len(v) > 500 for v in values):
            raise ValueError("Each requirement must have 1–500 characters")
        return list(dict.fromkeys(v.strip() for v in values))

    @field_validator("document_ids")
    @classmethod
    def unique_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


class RetrievalPlan(Contract):
    queries: list[str] = Field(min_length=1, max_length=13)
    facets: list[str] = Field(default_factory=list, max_length=12)
    trace: list[dict[str, object]] = Field(default_factory=list, max_length=13)

    @field_validator("queries", "facets")
    @classmethod
    def short_strings(cls, values: list[str]) -> list[str]:
        if any(not v.strip() or len(v) > 2000 for v in values):
            raise ValueError("Query/facet must have 1–2000 characters")
        return list(dict.fromkeys(v.strip() for v in values))


class QuestionAspect(Contract):
    text: str = Field(min_length=1, max_length=500)
    query: str = Field(min_length=1, max_length=1000)


class QuestionPlan(Contract):
    aspects: list[QuestionAspect] = Field(min_length=1, max_length=12)


class EvidenceGap(Contract):
    requirement_id: str | None = None
    reason: Literal["no_supported_evidence", "coverage_unassessed", "budget_exhausted"]
    scope: Literal["retrieved_context"] = "retrieved_context"


class GapCheck(Contract):
    gap_index: int = Field(ge=0, le=15)
    status: Literal["resolved", "unresolved", "unassessed", "irrelevant"]
    requirement_ids: list[str] = Field(default_factory=list, max_length=12)
    evidence_ids: list[str] = Field(default_factory=list, max_length=24)
    explanation: str = Field(min_length=1, max_length=700)


class RawEvidence(Contract):
    chunk_id: str = Field(min_length=1, max_length=100)
    quote: str = Field(default="", max_length=1400)
    unit_id: str | None = None
    requirement_ids: list[str] = Field(default_factory=list, max_length=12)
    relevance: str = Field(min_length=1, max_length=500)


class ReaderOutput(Contract):
    evidence: list[RawEvidence] = Field(default_factory=list, max_length=24)
    gaps: list[str] = Field(default_factory=list, max_length=8)


class Evidence(RawEvidence):
    id: str
    document_id: str
    document_title: str
    filename: str
    page: int | None
    section: str
    start: int
    end: int
    regions: list[SourceRegion] = Field(default_factory=list)


class RawClaim(Contract):
    text: str = Field(min_length=10, max_length=1000)
    evidence_ids: list[str] = Field(min_length=1, max_length=6)
    requirement_ids: list[str] = Field(default_factory=list, max_length=12)
    kind: Literal["source_finding", "grounded_recommendation"] = "source_finding"


class SynthesisOutput(Contract):
    claims: list[RawClaim] = Field(default_factory=list, max_length=16)
    gaps: list[str] = Field(default_factory=list, max_length=8)


class Claim(RawClaim):
    id: str


Verdict = Literal["supported", "contradicted", "insufficient"]


class ClaimCheck(Contract):
    claim_id: str
    verdict: Verdict
    explanation: str = Field(min_length=1, max_length=700)
    evidence_ids: list[str] = Field(default_factory=list, max_length=6)


class Requirement(Contract):
    id: str
    text: str


class CoverageCheck(Contract):
    requirement_id: str
    status: Literal["covered", "partial", "missing", "unassessed"]
    claim_ids: list[str] = Field(default_factory=list, max_length=16)
    explanation: str = Field(min_length=1, max_length=700)


FollowUpAction = Literal["retrieve", "read_context", "revise", "verify"]


class FollowUpRequest(Contract):
    action: FollowUpAction
    requirement_ids: list[str] = Field(min_length=1, max_length=12)
    claim_ids: list[str] = Field(default_factory=list, max_length=16)
    chunk_ids: list[str] = Field(default_factory=list, max_length=4)
    need: str = Field(min_length=8, max_length=700)


class AgentTask(FollowUpRequest):
    id: str
    requested_by: Literal["fact_checker", "supervisor", "orchestrator"] = "fact_checker"
    assigned_to: Literal["retrieval", "reader", "summarizer", "fact_checker"]
    status: Literal["pending", "running", "resolved", "unresolved"] = "pending"
    attempts: int = Field(default=0, ge=0)


class CoordinationDecision(Contract):
    action: Literal["retrieve", "read_context", "revise", "verify", "finish"]
    task_ids: list[str] = Field(default_factory=list, max_length=4)
    reason: str = Field(min_length=8, max_length=700)


class RetrievalAction(Contract):
    action: Literal["search", "finish"]
    queries: list[str] = Field(default_factory=list, max_length=4)
    mode: Literal["lexical", "hybrid"] = "hybrid"
    reason: str = Field(min_length=8, max_length=500)

    @field_validator("queries")
    @classmethod
    def bounded_queries(cls, values: list[str]) -> list[str]:
        if any(not q.strip() or len(q) > 1000 for q in values):
            raise ValueError("Each search query must contain 1-1000 characters")
        return list(dict.fromkeys(q.strip() for q in values))


class FactCheckOutput(Contract):
    checks: list[ClaimCheck] = Field(default_factory=list, max_length=16)
    coverage: list[CoverageCheck] = Field(default_factory=list, max_length=12)
    gap_checks: list[GapCheck] = Field(default_factory=list, max_length=16)
    requests: list[FollowUpRequest] = Field(default_factory=list, max_length=8)


class WrittenReport(Contract):
    title: str
    markdown: str
    warnings: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)


class ReportBlock(Contract):
    claim_id: str
    # The writer selects/orders verified claims. Claim text stays immutable after checking.


class ReportSection(Contract):
    heading: str = Field(min_length=1, max_length=120)
    blocks: list[ReportBlock] = Field(min_length=1, max_length=16)


class WriterOutput(Contract):
    title: str = Field(min_length=1, max_length=180)
    sections: list[ReportSection] = Field(default_factory=list, max_length=13)


WorkerRole = Literal["retrieval", "reader", "summarizer", "fact_checker", "writer"]
RuntimeRole = WorkerRole
# Request provenance also records code scheduling and retired historical roles.
TaskRequester = Literal[
    "retrieval", "reader", "summarizer", "fact_checker", "writer", "orchestrator", "supervisor"
]
TaskStatus = Literal["pending", "running", "waiting", "completed", "blocked", "cancelled"]


class TaskRequest(Contract):
    role: WorkerRole
    instruction: str = Field(min_length=8, max_length=1000)
    requirement_ids: list[str] = Field(default_factory=list, max_length=12)
    claim_ids: list[str] = Field(default_factory=list, max_length=16)
    chunk_ids: list[str] = Field(default_factory=list, max_length=12)
    depends_on: list[str] = Field(default_factory=list, max_length=16)


class RuntimeTask(TaskRequest):
    id: str
    requested_by: TaskRequester = "orchestrator"
    parent_id: str | None = None
    status: TaskStatus = "pending"
    message: str = ""


class ToolArguments(Contract):
    queries: list[str] = Field(default_factory=list, max_length=4)
    search_mode: Literal["lexical", "hybrid"] | None = None
    chunk_ids: list[str] = Field(default_factory=list, max_length=12)
    evidence_ids: list[str] = Field(default_factory=list, max_length=24)
    claim_ids: list[str] = Field(default_factory=list, max_length=16)
    task_id: str | None = None
    task: TaskRequest | None = None

    @field_validator("queries")
    @classmethod
    def query_limits(cls, queries: list[str]) -> list[str]:
        if any(not q.strip() or len(q) > 1000 for q in queries):
            raise ValueError("Queries must contain 1-1000 characters")
        return list(dict.fromkeys(q.strip() for q in queries))


OutputT = TypeVar("OutputT", bound=BaseModel)


class AgentDecision(Contract, Generic[OutputT]):
    action: Literal["use_tool", "request_task", "finish", "blocked"]
    tool: str | None = Field(default=None, max_length=60)
    arguments: ToolArguments | None = None
    result: OutputT | None = None
    reason: str = Field(min_length=8, max_length=500)


class SearchCompletion(Contract):
    chunk_ids: list[str] = Field(default_factory=list, max_length=12)


class AgentSession(Contract):
    task_id: str
    # Accept old persisted sessions without making Supervisor an executable role.
    role: RuntimeRole | Literal["supervisor"]
    steps: int = Field(default=0, ge=0)
    status: TaskStatus = "pending"
    observations: list[dict[str, object]] = Field(default_factory=list)
    message: str = ""


class AgentEvent(Contract):
    sequence: int
    agent: str
    status: Literal["running", "completed", "failed", "cancelled"]
    round: int = 1
    message: str
    elapsed_ms: float = 0
    details: dict[str, object] = Field(default_factory=dict)
    created_at: str


class AgentUsage(Contract):
    agent: str
    provider: str
    model: str
    models_used: list[str] = Field(default_factory=list)
    execution: Literal["local", "api", "demo"]
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: float = 0
    estimated_cost_usd: float | None = None


class ResearchResult(Contract):
    run_id: str
    question: str
    title: str
    markdown: str
    retrieved: list[RetrievedChunk]
    evidence: list[Evidence]
    claims: list[Claim]
    checks: list[ClaimCheck]
    gaps: list[str]
    events: list[AgentEvent]
    provider: str
    model: str
    embedding_model: str | None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: float
    rounds: int
    warnings: list[str] = Field(default_factory=list)
    created_at: str
    requirements: list[Requirement] = Field(default_factory=list)
    coverage: list[CoverageCheck] = Field(default_factory=list)
    reported_claim_ids: list[str] = Field(default_factory=list)
    stop_reason: str = "legacy_result"
    policy_version: str = "unclassified"
    gap_details: list[EvidenceGap] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    profile: str = "academic"
    cache_hits: int = 0
    agent_usage: list[AgentUsage] = Field(default_factory=list)
    estimated_cost_usd: float | None = None
    orchestration_mode: Literal["fixed", "adaptive", "autonomous"] = "fixed"
    agent_tasks: list[AgentTask] = Field(default_factory=list)
    task_graph: list[RuntimeTask] = Field(default_factory=list)
    agent_sessions: list[AgentSession] = Field(default_factory=list)


class RunRecord(Contract):
    scope_documents: list[Document] = Field(default_factory=list)
    corpus_revision: int | None = None
    id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    request: ResearchRequest
    created_at: str
    updated_at: str
    events: list[AgentEvent] = Field(default_factory=list)
    result: ResearchResult | None = None
    error: str | None = None
