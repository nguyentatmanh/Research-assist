import re
from hashlib import sha256

from research_assist.agents.prompts.reader import INSTRUCTION
from research_assist.domain.models import Evidence, RawEvidence, ReaderOutput, RetrievedChunk
from research_assist.domain.ports import LanguageModel
from research_assist.domain.source_units import source_units
from research_assist.domain.text import tokens


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
        lookup = {chunk.id: chunk for chunk in chunks}
        evidence, seen, rejected = [], set(), 0
        rejection_notes: list[str] = []
        for item in raw.evidence:
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
