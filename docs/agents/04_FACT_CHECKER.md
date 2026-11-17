# Fact Checker

Entry point: `FactCheckerAgent.assess`. Implementation: `src/research_assist/agents/`.

**Đầu vào:** Claims, evidence, requirements, candidate gaps.

**Đầu ra:** FactCheckOutput: checks, coverage, gap_checks.

| Bước | Trách nhiệm |
|---|---|
| Entailment | Provider đánh giá từng claim, điều kiện, số liệu và đề xuất dựa trên premise. |
| Structural policy | Kiểm tra ID, kết quả trùng/thiếu, số và phủ định trực tiếp; fail closed. |
| Coverage | Đánh giá từng yêu cầu. Với nhiều ý, claim phải mang requirement_id tương ứng mới được dùng cho coverage. |
| Gap reviews | Đánh giá từng candidate gap theo index, trả resolved/unresolved/unassessed và evidence IDs. |
| Consistency gate | domain/coverage.py đối chiếu gaps với coverage. Unreviewed/unresolved không được đồng thời giữ covered; chỉ tạo gap có scope retrieved_context. |

Không khẳng định nguồn đúng ngoài đời. Demo chỉ xác nhận exact extraction, không tự chấm semantic coverage là covered.

Ngân sách, retry, cache và provider được quản lý tại ModelRouter/HTTPLanguageModel; không có vòng gọi model vô hạn trong agent. Xem `tests/test_workflow_regressions.py` và các test agent tương ứng.
