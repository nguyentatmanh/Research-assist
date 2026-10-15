"""The shared language between agents. Identifiers are assigned by the application."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class SourceSection(Contract):
    text: str
    page: int | None = None
    label: str = "Văn bản"
    extraction: Literal["text", "ocr"] = "text"
    warnings: list[str] = Field(default_factory=list)


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
    document_ids: list[str] = Field(default_factory=list, max_length=50)
    top_k: int = Field(default=8, ge=2, le=12)
    language: Literal["vi", "en"] = "vi"
    style: Literal["overview", "comparison", "literature_review"] = "overview"
    retrieval_mode: Literal["lexical", "hybrid"] = "hybrid"
    max_rounds: int = Field(default=2, ge=1, le=2)
    profile: Literal["balanced", "academic"] | None = None
    use_cache: bool = True
    requirements: list[str] = Field(default_factory=list, max_length=6)

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
    queries: list[str] = Field(min_length=1, max_length=3)
    facets: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("queries", "facets")
    @classmethod
    def short_strings(cls, values: list[str]) -> list[str]:
        if any(not v.strip() or len(v) > 2000 for v in values):
            raise ValueError("Query/facet must have 1–2000 characters")
        return list(dict.fromkeys(v.strip() for v in values))


class RawEvidence(Contract):
    chunk_id: str = Field(min_length=1, max_length=100)
    quote: str = Field(min_length=20, max_length=1400)
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


class RawClaim(Contract):
    text: str = Field(min_length=10, max_length=1000)
    evidence_ids: list[str] = Field(min_length=1, max_length=6)


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


class FactCheckOutput(Contract):
    checks: list[ClaimCheck] = Field(default_factory=list, max_length=16)
    coverage: list[CoverageCheck] = Field(default_factory=list, max_length=6)


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
    sections: list[ReportSection] = Field(default_factory=list, max_length=6)


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
    policy_version: str = "2.0"
    profile: str = "academic"
    cache_hits: int = 0
    agent_usage: list[AgentUsage] = Field(default_factory=list)
    estimated_cost_usd: float | None = None


class RunRecord(Contract):
    id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    request: ResearchRequest
    created_at: str
    updated_at: str
    events: list[AgentEvent] = Field(default_factory=list)
    result: ResearchResult | None = None
    error: str | None = None
