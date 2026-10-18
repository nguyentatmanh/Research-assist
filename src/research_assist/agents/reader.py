import re
from hashlib import sha256

from research_assist.agents.prompts.reader import INSTRUCTION
from research_assist.domain.models import Evidence, RawEvidence, ReaderOutput, RetrievedChunk
from research_assist.domain.ports import LanguageModel
from research_assist.domain.source_units import source_units
from research_assist.domain.text import fold, tokens


def _asks_for_process(question: str) -> bool:
    return bool(
        re.search(
            r"\b(?:buoc|giai doan|quy trinh|trinh tu|steps?|stages?|process|procedure)\b",
            fold(question),
        )
    )


def _process_list_score(text: str) -> int:
    """Find an explicit sequence of actions, not a passing mention of a step."""
    folded = fold(text)
    if not re.search(r"\b(?:buoc|giai doan|steps?|stages?)\b", folded):
        return 0
    listed = len(re.findall(r"(?m)^\s*(?:[•*+-]|\d+[.)])\s+", text))
    if listed >= 2 and text.rstrip().endswith((".", ";", "!", "?")):
        return 10 + listed
    lead = re.search(r"\b(?:buoc|giai doan|steps?|stages?)\b[^:\n]{0,60}:\s*", folded)
    if lead and folded[lead.end() : lead.end() + 250].count(",") >= 2:
        return 5
    return 0


def quote_span(text: str, quote: str) -> tuple[int, int] | None:
    """Exact characters, with whitespace-only variation allowed. No fuzzy quotation."""
    pieces = re.split(r"\s+", quote.strip())
    if not pieces or len(quote.strip()) < 20:
        return None
    match = re.search(r"\s+".join(re.escape(piece) for piece in pieces), text)
    return match.span() if match else None


class ReaderAgent:
    name = "reader"

    def __init__(self, llm: LanguageModel) -> None:
        self.llm = llm

    async def run(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        requirements: list[str] | None = None,
    ) -> tuple[list[Evidence], list[str], list[str]]:
        if not chunks:
            return [], ["Không tìm thấy đoạn tài liệu liên quan đến câu hỏi."], []
        units = source_units(chunks)
        unit_lookup = {u["unit_id"]: u for u in units}
        raw = await self.llm.generate(
            task="read_evidence",
            instruction=(INSTRUCTION),
            payload={
                "question": question,
                "requirements": requirements or [],
                "source_units": units,
                "sources": [
                    {
                        "chunk_id": c.id,
                        "page": c.page,
                        "section": c.section,
                        "filename": c.filename,
                        "title": c.document_title,
                    }
                    for c in chunks
                ],
            },
            schema=ReaderOutput,
            fallback=lambda: self._extract(question, chunks),
        )
        selected = list(raw.evidence)
        supplemented = 0
        topic_terms = set(tokens(question)) - {
            "buoc",
            "giai",
            "doan",
            "quy",
            "trinh",
            "steps",
            "step",
            "stages",
            "stage",
            "process",
            "procedure",
            "chinh",
            "qua",
        }

        def process_candidate(unit: dict[str, str | int]) -> int:
            text = str(unit["text"])
            overlap = len(topic_terms & set(tokens(text)))
            if len(topic_terms) >= 2 and overlap < 2:
                return 0
            score = _process_list_score(text)
            return score + overlap if score else 0

        if _asks_for_process(question) and not any(
            process_candidate(unit_lookup[item.unit_id])
            for item in selected
            if item.unit_id in unit_lookup
            and unit_lookup[item.unit_id]["chunk_id"] == item.chunk_id
        ):
            ranked = sorted(
                (u for u in units if process_candidate(u)),
                key=process_candidate,
                reverse=True,
            )
            for unit in ranked[:2]:
                if len(selected) >= 24:
                    break
                selected.append(
                    RawEvidence(
                        chunk_id=str(unit["chunk_id"]),
                        unit_id=str(unit["unit_id"]),
                        quote="",
                        relevance="Nguồn liệt kê trực tiếp các bước hoặc giai đoạn.",
                        requirement_ids=["rq_1"] if len(requirements or []) == 1 else [],
                    )
                )
                supplemented += 1
        lookup = {chunk.id: chunk for chunk in chunks}
        evidence, seen, rejected = [], set(), 0
        rejection_notes: list[str] = []
        for item in selected:
            chunk = lookup.get(item.chunk_id)
            unit = unit_lookup.get(item.unit_id) if item.unit_id else None
            if item.unit_id:
                span = (
                    (int(unit["start"]), int(unit["end"]))
                    if (unit and unit["chunk_id"] == item.chunk_id)
                    else None
                )
            else:
                span = quote_span(chunk.text, item.quote) if chunk else None
            if chunk is None or span is None:
                rejected += 1
                rejection_notes.append(f"{item.chunk_id}: invalid_source_unit_or_quote")
                continue
            quote = chunk.text[span[0] : span[1]]
            if len(quote) > 1400:
                rejected += 1
                continue
            key = (chunk.document_id, chunk.page, quote)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                Evidence(
                    id="ev_"
                    + sha256(f"{chunk.id}:{span[0]}:{span[1]}:{quote}".encode()).hexdigest()[:24],
                    chunk_id=chunk.id,
                    quote=quote,
                    relevance=item.relevance,
                    unit_id=item.unit_id,
                    requirement_ids=[
                        rid
                        for rid in item.requirement_ids
                        if rid in {f"rq_{i + 1}" for i in range(len(requirements or []))}
                    ],
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    filename=chunk.filename,
                    page=chunk.page,
                    section=chunk.section,
                    start=chunk.start + span[0],
                    end=chunk.start + span[1],
                )
            )
        warnings = (
            [f"Reader: đã loại {rejected} trích dẫn không tìm thấy nguyên văn trong nguồn."]
            if rejected
            else []
        )
        if any("OCR" in e.section for e in evidence):
            warnings.append("Có bằng chứng trích từ OCR; hãy đối chiếu số liệu với ảnh trang gốc.")
        if supplemented:
            warnings.append(
                f"Reader: bổ sung {supplemented} cửa sổ nguồn liệt kê quy trình; "
                "các nhận định vẫn cần Fact Checker xác nhận."
            )
        warnings.extend(rejection_notes)
        return evidence, raw.gaps, warnings

    def _extract(self, question: str, chunks: list[RetrievedChunk]) -> ReaderOutput:
        query_terms = set(tokens(question))
        results = []
        for chunk in chunks:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?。])\s+|\n+", chunk.text)]
            candidates = [s for s in sentences if 30 <= len(s) <= 1000]
            ordered = sorted(candidates, key=lambda s: -len(query_terms & set(tokens(s))))
            for sentence in ordered[:2]:
                if not query_terms.intersection(tokens(sentence)):
                    continue
                results.append(
                    RawEvidence(
                        chunk_id=chunk.id,
                        quote=sentence,
                        relevance=(
                            "Trích xuất theo từ khóa ở chế độ demo; "
                            "chưa đánh giá ngữ nghĩa bằng LLM."
                        ),
                    )
                )
        return ReaderOutput(
            evidence=results[:20], gaps=[] if results else ["Không đủ bằng chứng theo từ khóa."]
        )
