# Pipeline nghiên cứu và bằng chứng — ghi chú của dự án

Đây là mô tả thiết kế của dự án, không phải kết quả thực nghiệm của một paper.

Retrieval Agent lập các truy vấn và gọi công cụ tìm kiếm để lấy đoạn tài liệu liên quan.
Reader Agent đọc đoạn truy xuất và lấy trích dẫn nguyên văn, giữ số trang của PDF.
Summarizer Agent tổng hợp các trích dẫn thành nhận định có danh sách bằng chứng.
Fact Checker Agent phân loại nhận định thành được hỗ trợ, mâu thuẫn hoặc chưa đủ bằng chứng.
Writer Agent tổ chức các nhận định đã được hỗ trợ thành báo cáo và gắn trích dẫn bằng mã nguồn.
BM25 tìm theo từ khóa, còn dense retrieval sử dụng embedding của mô hình đã huấn luyện để so sánh ngữ nghĩa.
Reciprocal Rank Fusion kết hợp các danh sách xếp hạng, không biến điểm retrieval thành xác suất đúng.
Pipeline cho phép tối đa hai vòng truy xuất khi thiếu bằng chứng, nhằm giới hạn chi phí và tránh lặp vô hạn.
Chế độ demo chỉ trích xuất theo từ khóa và chưa đánh giá entailment bằng LLM.
Nhận định được nguồn hỗ trợ không đồng nghĩa nguồn đó phản ánh đúng thực tế.
Tài liệu scan cần OCR trước khi lập chỉ mục, và PDF hai cột phức tạp cần kiểm tra lại thứ tự đọc.
