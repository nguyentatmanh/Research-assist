# RAG — ghi chú học tập

Đây là ghi chú do dự án biên soạn để demo, không phải toàn văn paper.
Nguồn tham khảo: Lewis và cộng sự, Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS 2020. https://arxiv.org/abs/2005.11401

RAG kết hợp mô hình sinh văn bản với nguồn tri thức bên ngoài được truy xuất theo câu hỏi.
Retriever tìm các đoạn liên quan, sau đó generator sử dụng những đoạn đó để tạo câu trả lời.
Trong nghiên cứu RAG gốc, mô hình sử dụng chỉ mục vector Wikipedia và một bộ truy xuất neural đã được huấn luyện.
RAG-Sequence dùng cùng nhóm đoạn trong toàn bộ quá trình sinh, còn RAG-Token có thể sử dụng đoạn khác nhau cho từng token.
Chất lượng RAG phụ thuộc vào việc truy xuất được nguồn liên quan và việc generator sử dụng đúng bằng chứng.
Một trích dẫn tồn tại không tự chứng minh rằng mọi nhận định trong câu trả lời đều được nguồn hỗ trợ.
