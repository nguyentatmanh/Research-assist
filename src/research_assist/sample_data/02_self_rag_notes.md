# Self-RAG — ghi chú học tập

Đây là ghi chú do dự án biên soạn để demo, không phải toàn văn paper.
Nguồn tham khảo: Asai và cộng sự, Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. https://arxiv.org/abs/2310.11511

Self-RAG hướng tới việc truy xuất theo nhu cầu và tự đánh giá văn bản được sinh.
Mô hình Self-RAG được huấn luyện để tạo các reflection token phục vụ quyết định truy xuất và phê bình câu trả lời.
Các khía cạnh phản tư bao gồm độ liên quan của đoạn truy xuất, mức hỗ trợ cho câu trả lời và tính hữu ích của đầu ra.
Khác với RAG luôn bổ sung một số đoạn cố định, Self-RAG có thể quyết định khi nào cần truy xuất trong quá trình sinh.
Việc ghép một agent kiểm tra vào RAG không đồng nghĩa đã tái hiện mô hình Self-RAG được huấn luyện bằng reflection token.
Trong dự án này, Fact Checker đánh giá nhận định so với bằng chứng; đây là một cách vận dụng ý tưởng kiểm tra, không phải mô hình Self-RAG gốc.
