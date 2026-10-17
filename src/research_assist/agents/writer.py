import re

from research_assist.agents.policies.verification import validate_checks
from research_assist.agents.prompts.writer import INSTRUCTION
from research_assist.domain.coverage import gap_messages, requirements_for
from research_assist.domain.models import (
    Claim,
    ClaimCheck,
    Evidence,
    EvidenceGap,
    ReportBlock,
    ReportSection,
    ResearchRequest,
    WriterOutput,
    WrittenReport,
)
from research_assist.domain.ports import LanguageModel


def plain_heading(value: str) -> str:
    return re.sub(r"[\r\n#<>\[\]`*]", " ", value).strip()


def escape_markdown(value: str) -> str:
    return re.sub(r"([\\`*_{}\[\]<>()#!|])", r"\\\1", value).replace("\n", " ")


class WriterAgent:
    name = "writer"

    def __init__(self, llm: LanguageModel) -> None:
        self.llm = llm

    async def run(
        self,
        request: ResearchRequest,
        claims: list[Claim],
        checks: list[ClaimCheck],
        evidence: list[Evidence],
        gaps: list[str],
    ) -> tuple[str, str, list[str]]:
        report = await self.compose(request, claims, checks, evidence, gaps)
        return report.title, report.markdown, report.warnings

    async def compose(
        self,
        request: ResearchRequest,
        claims: list[Claim],
        checks: list[ClaimCheck],
        evidence: list[Evidence],
        gaps: list[str],
        gap_details: list[EvidenceGap] | None = None,
    ) -> WrittenReport:
        checks = validate_checks(claims, evidence, checks)
        supported = {check.claim_id: check for check in checks if check.verdict == "supported"}
        verified = [c for c in claims if c.id in supported]
        en = request.language == "en"
        if not verified:
            title = "Insufficient evidence" if en else "Chưa đủ bằng chứng để trả lời"
            message = (
                "No supported claim could be established from the selected documents. "
                "Upload relevant sources or narrow the question."
                if en
                else "Chưa có nhận định nào đủ bằng chứng trong các tài liệu đã chọn. "
                "Hãy tải thêm tài liệu liên quan hoặc thu hẹp câu hỏi."
            )
            return WrittenReport(title=title, markdown=f"# {title}\n\n{message}\n")
        output = await self.llm.generate(
            task="write_report",
            instruction=(INSTRUCTION),
            payload={
                "question": request.question,
                "language": request.language,
                "style": request.style,
                "requirements": request.requirements,
                "verified_claims": [c.model_dump() for c in verified],
            },
            schema=WriterOutput,
            fallback=lambda: WriterOutput(
                title=("Research notes" if en else "Ghi chú nghiên cứu")
                + ": "
                + request.question[:140],
                sections=self._sections(verified, request),
            ),
        )
        lookup = {c.id: c for c in verified}
        ev_lookup = {e.id: e for e in evidence}
        cited: list[str] = []
        used, warnings = set(), []
        # Factual prose cannot bypass checking through a model-generated title or heading.
        title = "Research notes" if en else "Ghi chú nghiên cứu"
        headings = {
            "evidence": ("Evidence", "Bằng chứng"),
            "evidence from sources": ("Evidence", "Bằng chứng"),
            "bằng chứng từ tài liệu": ("Evidence", "Bằng chứng"),
            "grounded proposals": ("Grounded proposals", "Đề xuất từ bằng chứng"),
            "methods": ("Methods", "Phương pháp"),
            "results": ("Results", "Kết quả"),
            "comparison": ("Comparison", "So sánh"),
            "limitations": ("Limitations", "Hạn chế"),
        }
        headings.update({pair[1].casefold(): pair for pair in list(headings.values())})
        lines = [
            f"# {title}",
            "",
            ("**Question:** " if en else "**Câu hỏi:** ") + escape_markdown(request.question),
            "",
        ]
        selected_ids = {b.claim_id for section in output.sections for b in section.blocks}
        omitted = [c for c in verified if c.id not in selected_ids]
        if omitted:
            output.sections.append(
                ReportSection(
                    heading="Evidence from sources",
                    blocks=[ReportBlock(claim_id=c.id) for c in omitted],
                )
            )
        for section in output.sections:
            items = []
            for block in section.blocks:
                if block.claim_id not in lookup or block.claim_id in used:
                    if block.claim_id not in lookup:
                        warnings.append(
                            "Writer: đã loại một nhận định chưa được kiểm tra hoặc không tồn tại."
                        )
                    continue
                used.add(block.claim_id)
                claim = lookup[block.claim_id]
                markers = []
                for eid in supported[claim.id].evidence_ids:
                    if eid not in cited:
                        cited.append(eid)
                    markers.append(f"[{cited.index(eid) + 1}]")
                prefix = (
                    ("Proposal: " if en else "Đề xuất: ")
                    if claim.kind == "grounded_recommendation"
                    else ""
                )
                items.append(prefix + escape_markdown(claim.text) + " " + " ".join(markers))
            if items:
                pair = headings.get(section.heading.casefold())
                if section.heading in [r[:120] for r in request.requirements]:
                    pair = (section.heading, section.heading)
                if pair is None:
                    pair = ("Evidence", "Bằng chứng")
                    warnings.append("Writer: đã thay tiêu đề mục ngoài danh sách trung lập.")
                lines += [
                    "## " + plain_heading(pair[0 if en else 1]),
                    "",
                    *[f"- {item}" for item in items],
                    "",
                ]
        if not used:
            # On structural failure, render the immutable supported claims directly.
            warnings.append(
                "Writer không chọn nhận định hợp lệ; báo cáo dùng cấu trúc bằng chứng mặc định."
            )
            lines += ["## " + ("Evidence" if en else "Bằng chứng"), ""]
            for claim in verified:
                used.add(claim.id)
                markers = []
                for eid in supported[claim.id].evidence_ids:
                    if eid not in cited:
                        cited.append(eid)
                    markers.append(f"[{cited.index(eid) + 1}]")
                lines += ["- " + escape_markdown(claim.text) + " " + " ".join(markers)]
            lines.append("")
        lines += ["## " + ("Sources" if en else "Nguồn trích dẫn"), ""]
        for index, eid in enumerate(cited, 1):
            source = ev_lookup[eid]
            locator = f"p. {source.page}" if source.page is not None else source.section
            lines.append(
                f"[{index}] {escape_markdown(source.filename)} — {escape_markdown(locator)} "
                f"— `{source.chunk_id}`\n\n> {escape_markdown(source.quote)}\n"
            )
        if gap_details is not None:
            gaps = gap_messages(gap_details, requirements_for(request), request.language)
        elif gaps:
            # Legacy callers cannot inject unchecked negative factual statements.
            gaps = [
                (
                    "Some aspects remain unverified in retrieved evidence."
                    if en
                    else "Một số ý chưa được xác minh đầy đủ trong bằng chứng đã truy xuất."
                )
            ]
        if gaps:
            lines += ["## " + ("Evidence gaps" if en else "Khoảng trống bằng chứng"), ""]
            lines += ["- " + escape_markdown(gap) for gap in dict.fromkeys(gaps)]
            lines.append("")
        if self.llm.name == "demo":
            lines.append(
                "*Extractive demo: no LLM or semantic entailment check was used.*"
                if en
                else (
                    "*Chế độ demo trích xuất: không gọi LLM, "
                    "chưa kiểm tra entailment bằng mô hình.*"
                )
            )
        else:
            lines.append(
                "*Supported means supported by the supplied sources, "
                "not independently proven true.*"
                if en
                else (
                    "*Được hỗ trợ nghĩa là được nguồn cung cấp hỗ trợ; "
                    "không phải xác minh chân lý độc lập.*"
                )
            )
        return WrittenReport(
            title=title,
            markdown="\n".join(lines) + "\n",
            warnings=list(dict.fromkeys(warnings)),
            claim_ids=[c.id for c in verified if c.id in used],
        )

    @staticmethod
    def _sections(claims: list[Claim], request: ResearchRequest) -> list[ReportSection]:
        sections = []
        assigned: set[str] = set()
        for i, requirement in enumerate(request.requirements, 1):
            selected = [
                c for c in claims if f"rq_{i}" in c.requirement_ids and c.id not in assigned
            ]
            if selected:
                sections.append(
                    ReportSection(
                        heading=requirement[:120],
                        blocks=[ReportBlock(claim_id=c.id) for c in selected],
                    )
                )
                assigned.update(c.id for c in selected)
        rest = [c for c in claims if c.id not in assigned]
        if rest:
            sections.append(
                ReportSection(
                    heading="Evidence from sources",
                    blocks=[ReportBlock(claim_id=c.id) for c in rest],
                )
            )
        return sections
