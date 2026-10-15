import re
from hashlib import sha256

from research_assist.agents.prompts.reader import INSTRUCTION
from research_assist.domain.models import Evidence, RawEvidence, ReaderOutput, RetrievedChunk
from research_assist.domain.ports import LanguageModel
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
    ) -> tuple[list[Evidence], list[str], list[str]]:
        if not chunks:
            return [], ["Không tìm thấy đoạn tài liệu liên quan đến câu hỏi."], []
        raw = await self.llm.generate(
            task="read_evidence",
            instruction=(INSTRUCTION),
            payload={"question": question, "chunks": [c.model_dump() for c in chunks]},
            schema=ReaderOutput,
            fallback=lambda: self._extract(question, chunks),
        )
        lookup = {chunk.id: chunk for chunk in chunks}
        evidence, seen, rejected = [], set(), 0
        for item in raw.evidence:
            chunk = lookup.get(item.chunk_id)
            span = quote_span(chunk.text, item.quote) if chunk else None
            if chunk is None or span is None:
                rejected += 1
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
