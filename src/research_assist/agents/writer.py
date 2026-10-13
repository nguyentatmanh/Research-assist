import re

from research_assist.domain.models import (
    Claim,
    ClaimCheck,
    Evidence,
    ReportBlock,
    ReportSection,
    ResearchRequest,
    WriterOutput,
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
            return title, f"# {title}\n\n{message}\n", []
        output = await self.llm.generate(
            task="write_report",
            instruction=(
                "Organize a research report in the requested language/style. Return a concise "
                "title and 1–5 thematic sections. Each block contains ONLY a supplied verified "
                "claim_id. Select and order relevant claims; do NOT rewrite them, create new "
                "factual assertions in headings, or invent references. Use neutral topic "
                "headings. For comparison group by comparison dimensions; for literature "
                "review group by themes, methods, and limitations when supported."
            ),
            payload={
                "question": request.question,
                "language": request.language,
                "style": request.style,
                "verified_claims": [c.model_dump() for c in verified],
            },
            schema=WriterOutput,
            fallback=lambda: WriterOutput(
                title=("Research notes" if en else "Ghi chú nghiên cứu")
                + ": "
                + request.question[:140],
                sections=[
                    ReportSection(
                        heading="Evidence from sources" if en else "Bằng chứng từ tài liệu",
                        blocks=[ReportBlock(claim_id=c.id) for c in verified],
                    )
                ],
            ),
        )
        lookup = {c.id: c for c in verified}
        ev_lookup = {e.id: e for e in evidence}
        cited: list[str] = []
        used, warnings = set(), []
        title = plain_heading(output.title) or ("Research notes" if en else "Ghi chú nghiên cứu")
        lines = [
            f"# {title}",
            "",
            ("**Question:** " if en else "**Câu hỏi:** ") + escape_markdown(request.question),
            "",
        ]
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
                items.append(escape_markdown(claim.text) + " " + " ".join(markers))
            if items:
                lines += [
                    "## " + (plain_heading(section.heading) or "Evidence"),
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
        return title, "\n".join(lines) + "\n", list(dict.fromkeys(warnings))
