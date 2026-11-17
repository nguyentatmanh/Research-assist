# Retrieval

Entry point: `RetrievalAgent.prepare / run`. Implementation: `src/research_assist/agents/`.

**Đầu vào:** ResearchRequest, QuestionPlan, pending requirement IDs.

**Đầu ra:** RetrievalPlan, RetrievedChunk[], warnings.

| Bước | Trách nhiệm |
|---|---|
| Planning | Một lần mỗi phiên; provider cho câu phức tạp, local cho câu đơn giản/demo. Requirements người dùng được giữ nguyên; query có thể dịch sang tiếng Anh. |
| Scope | Nhận document_ids đã chốt; không tự mở rộng sang kho khác. |
| Candidate lanes | Mỗi query lấy sâu tới candidate limit; phân bổ xen kẽ các chunk khác nhau trước khi rerank theo query. Tổng số chunk rerank không vượt ngân sách; giữ query gốc. |
| Selection | Round-robin theo rank; khử trùng ID; top_k. Vòng sau ưu tiên các ý còn thiếu. |
| Context | HybridSearch mở rộng một neighbor liên tục cùng mục/trang; không thay vector của child. |

domain/planning.py, infrastructure/search.py. Trace có query, thứ hạng trước/sau rerank và danh sách context cuối. Điểm xếp hạng không phải confidence.

Ngân sách, retry, cache và provider được quản lý tại ModelRouter/HTTPLanguageModel; không có vòng gọi model vô hạn trong agent. Xem `tests/test_workflow_regressions.py`, `tests/test_review_regressions.py` và các test agent tương ứng.
