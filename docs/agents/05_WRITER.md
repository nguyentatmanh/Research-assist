# Writer

Entry point: `WriterAgent.compose`. Implementation: `src/research_assist/agents/`.

**Đầu vào:** Request/requirements, claims/checks, evidence, typed gaps.

**Đầu ra:** WrittenReport: Markdown, claim IDs, warnings.

| Bước | Trách nhiệm |
|---|---|
| Guard | Xác thực lại checks ngay tại boundary; chỉ lấy supported claims. |
| Structure | Balanced nhóm claim theo requirement bằng Python; academic dùng provider chọn thứ tự/section. |
| Completeness | Server thêm lại claim đã kiểm tra mà model bỏ sót khi tổ chức báo cáo. |
| Rendering | Giữ nguyên text sau kiểm chứng, escape Markdown, dựng citation và quote trên server. Đề xuất được gắn nhãn. |
| Gaps | Chỉ dùng thông báo do server dựng từ typed gaps; legacy string gaps không được in nguyên văn thành khẳng định phủ định. |

Fallback tiếng Việt nằm trong danh sách heading hợp lệ; không còn tự sinh cảnh báo tiêu đề. Writer không tải model .pt riêng trong profile balanced.

Ngân sách, retry, cache và provider được quản lý tại ModelRouter/HTTPLanguageModel; không có vòng gọi model vô hạn trong agent. Xem `tests/test_workflow_regressions.py` và các test agent tương ứng.
