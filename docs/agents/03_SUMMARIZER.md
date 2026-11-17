# Summarizer

Entry point: `SummarizerAgent.run`. Implementation: `src/research_assist/agents/`.

**Đầu vào:** Evidence ledger, question, language, requirements.

**Đầu ra:** Claim[], candidate gaps, warnings.

| Bước | Trách nhiệm |
|---|---|
| Synthesis | Tối đa 16 claim nguyên tử, trải theo các ý có evidence. |
| Kinds | source_finding phải được nguồn trực tiếp hỗ trợ; grounded_recommendation chỉ khi người dùng yêu cầu đề xuất, có điều kiện và premise được dẫn. |
| Mapping | Claim mang evidence_ids và requirement_ids. Gắn ID không thay cho entailment. |
| Guards | Loại ID evidence không tồn tại, khử trùng text, gắn claim ID xác định từ text/evidence. |
| Retention | Orchestrator có thể giữ supported claim vòng trước khi evidence còn trong pool; Checker kiểm tra lại. |

Instruction ở agents/prompts/summarizer.py. Không tạo phần trăm, tác giả, kết quả thực nghiệm hoặc lời bảo đảm không có trong nguồn.

Ngân sách, retry, cache và provider được quản lý tại ModelRouter/HTTPLanguageModel; không có vòng gọi model vô hạn trong agent. Xem `tests/test_workflow_regressions.py` và các test agent tương ứng.
