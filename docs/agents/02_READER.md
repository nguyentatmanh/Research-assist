# Reader

Entry point: `ReaderAgent.run`. Implementation: `src/research_assist/agents/`.

**Đầu vào:** Question, retrieved chunks, requirements.

**Đầu ra:** Evidence[], candidate gaps, warnings.

| Bước | Trách nhiệm |
|---|---|
| Units | domain/source_units.py tạo các cửa sổ nguyên văn tối đa 869 ký tự khi ghép phần dư ngắn với ID xác định từ chunk/offset/text. |
| Selection | Provider chọn chunk_id + unit_id và requirement_ids; không cần gõ lại quote. |
| Reconstruction | Server lấy substring chính xác, gắn nguồn/trang/offset. Quote trực tiếp từ adapter cũ vẫn phải qua exact/whitespace-only check. |
| Validation | ID lạ, unit không thuộc chunk hoặc quote sai bị loại; cảnh báo ghi chunk ID và reason. Không fuzzy matching. |
| Output | Khử trùng bằng document/page/quote. Candidate gaps chỉ là điều cần kiểm tra, không là kết luận về cả tài liệu. |

API/provider đọc nguồn; việc dựng quote và kiểm tra offset chạy Python local. Không tải một model Reader .pt riêng.

Ngân sách, retry, cache và provider được quản lý tại ModelRouter/HTTPLanguageModel; không có vòng gọi model vô hạn trong agent. Xem `tests/test_workflow_regressions.py` và các test agent tương ứng.
