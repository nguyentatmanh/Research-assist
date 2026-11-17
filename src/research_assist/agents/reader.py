import json
import re
from hashlib import sha256

from research_assist.agents.prompts.reader import INSTRUCTION
from research_assist.domain.comparison import measured_metrics, metric_names
from research_assist.domain.models import (
    Evidence,
    FollowUpRequest,
    RawEvidence,
    ReaderOutput,
    RetrievedChunk,
)
from research_assist.domain.ports import LanguageModel
from research_assist.domain.provenance import evidence_source_hash, intersect_regions
from research_assist.domain.source_units import source_units
from research_assist.domain.text import bm25_tokens, extract_literal_anchors, fold, tokens

_READ_TASK = "read_evidence"


def _rank_source_units(
    units: list[dict[str, str | int]],
    query: str,
    chunks: list[RetrievedChunk],
    cover_documents: bool,
) -> list[dict[str, str | int]]:
    # Keep the narrower branch terms ahead of the original question. Treat
    # English–German and English-to-German as sharing the same language names.
    terms = set(bm25_tokens(re.sub(r"[-‐‑–_]", " ", query)))
    anchors = extract_literal_anchors(query)
    metrics = metric_names(query) if cover_documents else set()
    by_id = {c.id: c for c in chunks}
    ranked = []
    seen = set()
    for unit in units:
        chunk = by_id[str(unit["chunk_id"])]
        span = (
            chunk.document_id,
            chunk.page,
            chunk.start + int(unit["start"]),
            chunk.start + int(unit["end"]),
        )
        if span in seen:
            continue
        seen.add(span)
        text = str(unit["text"])
        overlap = len(terms.intersection(bm25_tokens(re.sub(r"[-‐‑–_]", " ", text))))
        metric_matches = len(metrics & metric_names(text)) if metrics else 0
        has_measurement = bool(metric_matches and metrics & measured_metrics(text))
        ranked.append(
            (
                len(anchors & extract_literal_anchors(text)),
                has_measurement,
                metric_matches,
                overlap,
                unit,
            )
        )
    ranked.sort(key=lambda item: item[:4], reverse=True)
    ordered = [unit for _, _, _, _, unit in ranked]
    if not cover_documents:
        return ordered
    by_document: dict[str, list[dict[str, str | int]]] = {}
    for unit in ordered:
        doc_id = by_id[str(unit["chunk_id"])].document_id
        by_document.setdefault(doc_id, []).append(unit)
    reserved = [lane[i] for i in range(2) for lane in by_document.values() if i < len(lane)]
    reserved_ids = {unit["unit_id"] for unit in reserved}
    return reserved + [unit for unit in ordered if unit["unit_id"] not in reserved_ids]


def _missing_metric_windows(
    units: list[dict[str, str | int]],
    chunks: list[RetrievedChunk],
    question: str,
    selected: list[RawEvidence],
) -> list[RawEvidence]:
    requested = metric_names(question)
    if not requested:
        return []

    def reports_metric(text: str) -> bool:
        return bool(requested.intersection(measured_metrics(text)))

    by_id = {c.id: c.document_id for c in chunks}
    by_unit = {unit["unit_id"]: unit for unit in units}
    represented: set[str] = set()
    for item in selected:
        if item.unit_id is None:
            continue
        unit = by_unit.get(item.unit_id)
        if unit and unit["chunk_id"] == item.chunk_id and reports_metric(str(unit["text"])):
            represented.add(by_id[item.chunk_id])
    candidates = []
    for unit in units:
        chunk_id = str(unit["chunk_id"])
        doc_id = by_id[chunk_id]
        if doc_id in represented or not reports_metric(str(unit["text"])):
            continue
        candidates.append(
            RawEvidence(
                chunk_id=chunk_id,
                unit_id=str(unit["unit_id"]),
                quote="",
                relevance="Đoạn có tên thước đo được hỏi và số liệu để kiểm chứng.",
            )
        )
        represented.add(doc_id)
        if len(candidates) + len(selected) >= 24:
            break
    return candidates


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


def _missing_literal_anchor(question: str, chunks: list[RetrievedChunk]) -> bool:
    """Extractive fallback needs literal support for explicit identifiers and units."""
    # Only digit-bearing identifiers, mixed-case units and percentages need a
    # literal match. Ordinary hyphenated prose can use spaces in the source.
    anchors = re.findall(
        r"(?<!\w)(?:[A-Za-z]+(?:[-‐‑–][A-Za-z\d]+)*[-‐‑–]\d+[A-Za-z\d-]*|"
        r"[a-z]+[A-Z][A-Za-z]*|"
        r"[A-Za-z]{2,}\d+[A-Za-z\d.-]*|\d+(?:[.,]\d+)?%)(?!\w)",
        question,
    )
    if not anchors:
        return False

    def canonical(value: str) -> str:
        return re.sub(r"[-‐‑–]\s*", "-", fold(value))

    context = canonical("\n".join(c.text for c in chunks))
    return any(
        not re.search(r"(?<!\w)" + re.escape(canonical(a)) + r"(?!\w)", context) for a in anchors
    )


class ReaderAgent:
    name = "reader"

    def __init__(self, llm: LanguageModel) -> None:
        self.llm = llm

    async def run(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        requirements: list[str] | None = None,
        requests: list[FollowUpRequest] | None = None,
        *,
        cover_documents: bool = False,
    ) -> tuple[list[Evidence], list[str], list[str]]:
        if not chunks:
            return [], ["Không tìm thấy đoạn tài liệu liên quan đến câu hỏi."], []
        units = source_units(chunks)
        sources = [
            {
                "chunk_id": c.id,
                "page": c.page,
                "section": c.section,
                "filename": c.filename,
                "title": c.document_title,
            }
            for c in chunks
        ]
        base = {
            "question": question,
            "requirements": requirements or [],
            "reading_requests": [r.model_dump() for r in (requests or [])],
        }
        if cover_documents:
            base["evidence_goal"] = (
                "Extract the task and metric reported by each source independently. "
                "Return those quotes even if only one source is currently available. "
                "Leave the question of comparability to the synthesis step."
            )

        def payload_for(windows: list[dict[str, str | int]]) -> dict[str, object]:
            ids = {u["chunk_id"] for u in windows}
            return {
                **base,
                "source_units": windows,
                "sources": [source for source in sources if source["chunk_id"] in ids],
            }

        limit = getattr(self.llm, "input_limits", None)
        limits = limit(_READ_TASK, INSTRUCTION, ReaderOutput) if limit else None
        budget_warnings: list[str] = []
        if limits:
            max_chars, max_bytes = limits

            def fits(windows: list[dict[str, str | int]]) -> bool:
                serialized = json.dumps(payload_for(windows), ensure_ascii=False)
                return len(serialized) <= max_chars and (
                    max_bytes is None or len(serialized.encode()) <= max_bytes
                )

            if not fits(units):
                queries = [r.need for r in requests or []] or requirements or [question]
                ranked = _rank_source_units(units, " ".join(queries), chunks, cover_documents)
                low, high = 0, len(ranked)
                while low < high:
                    middle = (low + high + 1) // 2
                    if fits(ranked[:middle]):
                        low = middle
                    else:
                        high = middle - 1
                if not low:
                    reason = "Ngân sách Reader không đủ để gửi một đoạn nguồn."
                    return [], [reason], [reason]
                budget_warnings.append(
                    f"Reader: chọn {low}/{len(units)} đoạn nguồn để vừa ngân sách của tác vụ."
                )
                units = ranked[:low]
        raw = await self.llm.generate(
            task=_READ_TASK,
            instruction=INSTRUCTION,
            payload=payload_for(units),
            schema=ReaderOutput,
            fallback=lambda: self._extract(question, chunks),
        )
        if cover_documents and len(raw.evidence) < 24:
            ranked = _rank_source_units(units, question, chunks, cover_documents)
            supplemental = _missing_metric_windows(ranked, chunks, question, raw.evidence)
            if supplemental:
                raw = raw.model_copy(update={"evidence": [*raw.evidence, *supplemental]})
        evidence, gaps, warnings = self.validate_output(
            question,
            chunks,
            raw,
            requirements,
            windows=units,
        )
        return evidence, gaps, budget_warnings + warnings

    def validate_output(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        raw: ReaderOutput,
        requirements: list[str] | None = None,
        *,
        windows: list[dict[str, str | int]] | None = None,
    ) -> tuple[list[Evidence], list[str], list[str]]:
        units = source_units(chunks) if windows is None else windows
        unit_lookup = {u["unit_id"]: u for u in units}
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
            for candidate_unit in ranked[:2]:
                if len(selected) >= 24:
                    break
                selected.append(
                    RawEvidence(
                        chunk_id=str(candidate_unit["chunk_id"]),
                        unit_id=str(candidate_unit["unit_id"]),
                        quote="",
                        relevance="Nguồn liệt kê trực tiếp các bước hoặc giai đoạn.",
                        requirement_ids=["rq_1"] if len(requirements or []) == 1 else [],
                    )
                )
                supplemented += 1
        sent_ids = {u["chunk_id"] for u in units}
        lookup = {chunk.id: chunk for chunk in chunks if chunk.id in sent_ids}
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
            key = (chunk.document_id, chunk.page, chunk.start + span[0], chunk.start + span[1])
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
                    regions=intersect_regions(
                        chunk.regions, chunk.start + span[0], chunk.start + span[1]
                    ),
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
        for item_evidence in evidence:
            item_evidence.source_hash = evidence_source_hash(item_evidence)
        return evidence, raw.gaps, warnings

    def _extract(self, question: str, chunks: list[RetrievedChunk]) -> ReaderOutput:
        if _missing_literal_anchor(question, chunks):
            return ReaderOutput(
                evidence=[],
                gaps=["Thiếu bằng chứng cho định danh hoặc đơn vị được hỏi."],
            )
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
