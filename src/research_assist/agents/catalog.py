"""Machine-readable contracts used by the API and architecture documentation."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AgentSpec:
    role: str
    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    stages: tuple[str, ...]
    tools: tuple[str, ...]
    guard: str


SPECS = (
    AgentSpec(
        "retrieval",
        "Tìm đoạn tài liệu liên quan",
        ("ResearchRequest", "evidence gaps"),
        ("RetrievalPlan", "RetrievedChunk[]"),
        ("Lập truy vấn", "Giữ câu hỏi gốc", "BM25 + cosine + RRF", "Rerank", "Top K"),
        ("LanguageModel", "SearchTool", "Reranker"),
        "Luôn giữ phạm vi tài liệu đã chọn.",
    ),
    AgentSpec(
        "reader",
        "Trích bằng chứng có vị trí nguồn",
        ("question", "RetrievedChunk[]"),
        ("Evidence[]", "gaps", "warnings"),
        ("Trích dẫn", "Kiểm tra nguyên văn", "Khử trùng", "Gắn ID và offset"),
        ("LanguageModel",),
        "Loại quote không tồn tại và chunk_id lạ.",
    ),
    AgentSpec(
        "summarizer",
        "Tổng hợp nhận định nguyên tử",
        ("question", "Evidence[]", "language"),
        ("Claim[]", "gaps", "warnings"),
        ("Tổng hợp", "Kiểm tra evidence ID", "Khử trùng", "Gắn claim ID"),
        ("LanguageModel",),
        "Không thêm nhận định để đủ số lượng; giữ phạm vi nguồn.",
    ),
    AgentSpec(
        "fact_checker",
        "Kiểm chứng nhận định và độ đầy đủ",
        ("Claim[]", "Evidence[]", "Requirement[]"),
        ("ClaimCheck[]", "CoverageCheck[]"),
        ("Đánh giá entailment và yêu cầu", "Kiểm tra ID", "Kiểm tra số", "Xác thực coverage"),
        ("LanguageModel",),
        "Thiếu hoặc trùng kết quả: insufficient/unassessed.",
    ),
    AgentSpec(
        "writer",
        "Xuất báo cáo có nguồn",
        ("Claim[]", "ClaimCheck[]", "Evidence[]"),
        ("WrittenReport",),
        ("Kiểm tra lại claim", "Chọn và sắp xếp", "Heading trung lập", "Dựng trích dẫn"),
        ("LanguageModel",),
        "Chỉ xuất claim đã được hỗ trợ; không sửa nội dung sau kiểm chứng.",
    ),
)


def public_specs() -> list[dict[str, object]]:
    return [asdict(spec) for spec in SPECS]
