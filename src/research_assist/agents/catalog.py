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
        (
            "Phân rã câu hỏi một lần",
            "Tự chọn query, quan sát và đổi chiến lược",
            "Truy xuất theo từng ý",
            "BM25 + cosine + RRF",
            "Rerank theo truy vấn",
            "Phân bổ Top K + mở rộng ngữ cảnh",
        ),
        ("LanguageModel", "SearchTool", "Reranker", "search", "inspect_sources"),
        "Luôn giữ phạm vi tài liệu đã chọn.",
    ),
    AgentSpec(
        "reader",
        "Trích bằng chứng có vị trí nguồn",
        ("question", "RetrievedChunk[]"),
        ("Evidence[]", "gaps", "warnings"),
        (
            "Tạo source units nguyên văn",
            "Model chọn ID",
            "Khôi phục trích dẫn trên server",
            "Kiểm tra ID/offset và khử trùng",
        ),
        ("LanguageModel", "read_units", "read_context", "read_section", "request_task"),
        "Loại quote không tồn tại và chunk_id lạ.",
    ),
    AgentSpec(
        "summarizer",
        "Tổng hợp nhận định nguyên tử",
        ("question", "Evidence[]", "language"),
        ("Claim[]", "gaps", "warnings"),
        ("Tổng hợp", "Kiểm tra evidence ID", "Khử trùng", "Gắn claim ID"),
        ("LanguageModel", "inspect_evidence", "inspect_claims", "request_task"),
        "Không thêm nhận định để đủ số lượng; giữ phạm vi nguồn.",
    ),
    AgentSpec(
        "fact_checker",
        "Kiểm chứng nhận định và độ đầy đủ",
        ("Claim[]", "Evidence[]", "Requirement[]"),
        ("ClaimCheck[]", "CoverageCheck[]", "FollowUpRequest[]"),
        (
            "Đánh giá từng claim và ý",
            "Đánh giá khoảng trống",
            "Kiểm tra ID/số",
            "Đối chiếu coverage với gaps",
        ),
        ("LanguageModel", "inspect_evidence", "inspect_claims", "check_numbers", "request_task"),
        "Thiếu hoặc trùng kết quả: insufficient/unassessed.",
    ),
    AgentSpec(
        "writer",
        "Xuất báo cáo có nguồn",
        ("Claim[]", "ClaimCheck[]", "Evidence[]"),
        ("WrittenReport",),
        ("Kiểm tra lại claim", "Chọn và sắp xếp", "Heading trung lập", "Dựng trích dẫn"),
        ("LanguageModel", "inspect_claims", "inspect_progress", "request_task"),
        "Chỉ xuất claim đã được hỗ trợ; không sửa nội dung sau kiểm chứng.",
    ),
)


def public_specs() -> list[dict[str, object]]:
    return [asdict(spec) for spec in SPECS]
