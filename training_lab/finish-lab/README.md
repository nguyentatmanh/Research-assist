# 🚀 Danh Sách Notebook Huấn Luyện Đã Hoàn Thiện (Finish Lab)

Thư mục này chứa toàn bộ các Jupyter Notebooks đã được cấu hình tối ưu để huấn luyện trực tiếp trên **Google Colab** (GPU T4 miễn phí) và **Kaggle** (Dual Tesla T4 / P100).

---

## 📋 Chi tiết các Notebook:

### 1. `colab_agent1-agent2-agent4.ipynb` (Google Colab T4)
- **Agent 1 (Router & Planner)**: Tinh chỉnh mô hình `Qwen2.5-7B-Instruct` bằng QLoRA ($r=64$), huấn luyện kỹ năng phân rã truy vấn, lập kế hoạch và sinh JSON chuẩn schema.
- **Agent 2 (Document Reader)**: Huấn luyện `microsoft/layoutlmv3-base` với Focal Loss để nhận diện layout văn bản, bảng biểu và công thức toán học từ tài liệu PDF.
- **Agent 4 (Fact-Checker NLI)**: Tinh chỉnh đầu phân loại NLI trên `BAAI/bge-reranker-v2-m3` kết hợp R-Drop để kiểm định độ trung thực (groundedness) của từng nhận định.

### 2. `colab_agent3.ipynb` (Google Colab T4)
- **Agent 3 (Academic Summarizer & Critic)**: Huấn luyện kiến trúc Dual-QLoRA trên `Qwen2.5-7B-Instruct`:
  - **Generator SFT + DPO**: Tóm tắt có trích dẫn, áp dụng DPO phạt nặng các lỗi bịa đặt thông tin.
  - **Critic**: Tự động đánh giá và phản biện bản tóm tắt trước khi bàn giao cho Fact-Checker.

### 3. `kaggle_agent5.ipynb` (Kaggle GPU Dual T4 / P100)
- **Agent 5 (Academic Report Writer)**: Huấn luyện `Qwen2.5-7B-Instruct` với kỹ thuật **DoRA (Weight-Decomposed Low-Rank Adaptation)** và **rsLoRA** để biên soạn báo cáo học thuật hoàn chỉnh chuẩn IEEE từ các nhận định đã được kiểm chứng.

### 4. `kaggle_embedding-reranker.ipynb` (Kaggle Dual Tesla T4)
- **Dense Retriever**: Huấn luyện LoRA trên `BAAI/bge-m3` với `CachedMultipleNegativesRankingLoss` và hard negatives đa lĩnh vực (Y sinh, Tài chính, Khoa học Kỹ thuật).
- **Cross-Encoder Reranker**: Huấn luyện LoRA trên `BAAI/bge-reranker-v2-m3` theo tiêu chuẩn CoREB để xếp hạng lại độ phù hợp của các đoạn văn bản.

---

## 📌 Hướng dẫn Lưu Trữ & Tải Lên Google Drive
Sau khi chạy hoàn tất các notebook trên:
1. Nén toàn bộ thư mục adapter/weights thành tệp `final_weights_bundle.zip` (hoặc từng thư mục riêng).
2. Tải tệp lên Google Drive cá nhân của bạn và bật quyền chia sẻ (Anyone with the link can view).
3. Cập nhật đường dẫn / Folder ID Google Drive của bạn vào phần [V. ĐỒNG BỘ HÓA ĐÁM MÂY GOOGLE DRIVE](../README.md#v-đồng-bộ-hóa-đám-mây-google-drive--liên-kết-suy-luận).
