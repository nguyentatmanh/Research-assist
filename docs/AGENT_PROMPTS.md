# Prompt các agent

## Nguồn prompt và vai trò

| Thành phần | Tệp | Trách nhiệm |
|---|---|---|
| Planner / Retrieval | `src/research_assist/domain/planning.py` | Tách đúng yêu cầu, tạo truy vấn giữ tên bài, thực thể, thuật ngữ tiếng Việt và ràng buộc |
| Reader | `src/research_assist/agents/prompts/reader.py` | Chọn cửa sổ nguồn đủ ngữ cảnh; không tự viết lại trích dẫn |
| Summarizer | `src/research_assist/agents/prompts/summarizer.py` | Tạo nhận định ngắn, kiểm chứng riêng được, mỗi nhận định có nguồn phù hợp |
| Fact Checker | `src/research_assist/agents/prompts/fact_checker.py` | Kiểm tra ý nghĩa, độ đáp ứng và khoảng trống bằng chứng riêng biệt |
| Writer | `src/research_assist/agents/prompts/writer.py` | Sắp xếp ID nhận định đã được xác nhận, không sinh nội dung thực tế mới |
| Chỉ dẫn chung | `src/research_assist/infrastructure/llm.py` | Xác định nhiệm vụ từ câu hỏi trong phạm vi vai trò, giữ schema và chống chỉ dẫn nhúng trong tài liệu |

`agents/prompts/retrieval.py` xuất lại prompt Planner đang dùng, không còn giữ một prompt
khác không được gọi. Retrieval tìm kiếm và reranking vẫn là thao tác bằng code/model;
Planner chỉ được gọi khi câu hỏi cần phân rã. Cohere Rerank không nhận prompt hội thoại.
Trong profile balanced, Writer dùng bố cục local; prompt Writer có tác dụng khi chạy model.

## Các điểm đã chỉnh

- Planner không bắt buộc thay truy vấn tiếng Việt bằng tiếng Anh; giữ thuật ngữ đặc thù và
  không tự thêm yêu cầu ngoài câu hỏi.
- Reader lấy đủ danh sách bước, tiêu đề/cột/hàng của bảng khi có trong các đơn vị nguồn;
  giữ câu phủ định, ngoại lệ và bằng chứng trái chiều. Không suy đoán ô bảng bị mất.
- Summarizer đọc nội dung trích dẫn thay vì coi nhãn liên quan của Reader là bằng chứng;
  tách nhận định, giữ thực thể/đơn vị/điều kiện, không tự tính số liệu chưa có trong nguồn.
- Fact Checker chấp nhận bản dịch và diễn đạt lại đúng nghĩa, nhưng phải kiểm tra toàn bộ
  nhận định theo chính nguồn được dẫn; không tự lấy nguồn khác để cứu trích dẫn sai.
  Lý do bác bỏ cần chỉ rõ phần thiếu hoặc sai. Chấm độ đủ và kiểm tra gap tách riêng.
- Writer đặt đáp án trực tiếp trước, giữ thứ tự bước và không bỏ qua hạn chế liên quan.

## Cách kiểm tra chất lượng

Giữ cùng model, cấu hình, PDF và câu hỏi khi so sánh các phiên bản prompt. Kiểm tra cả
đáp án lẫn nguồn: bản dịch "encoder có 6 lớp" từ nguồn tiếng Anh về encoder là hợp lệ;
đổi thành decoder không hợp lệ dù cùng số 6. Danh sách bị cắt không chứng minh đầy đủ
quy trình; số trong bảng chỉ có nghĩa khi ánh xạ đúng hàng/cột/điều kiện.

Các guard về ID, số liệu, trích dẫn nguyên văn và Writer vẫn chạy sau phản hồi model.
Kiểm thử giả lập xác nhận các guard và hợp đồng đầu ra, không chứng minh model thật sẽ
hiểu đúng mọi tài liệu. Không dùng số ca test qua hoặc điểm tự chấm làm tỷ lệ chính xác.

Khởi động lại backend để nạp prompt mới. Cache model chứa nội dung prompt trong khóa nên
những prompt đã sửa không dùng lại phản hồi cũ. Không cần rebuild PDF hoặc embedding.
