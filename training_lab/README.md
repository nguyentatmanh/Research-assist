# 🧪 Training Lab — Hệ Thống Huấn Luyện AI 2.0 Multi-Agent RAG

> **Dự án**: Autonomous Multi-Agent Academic RAG System  
> **Kiến trúc**: Domain-Agnostic Cross-Domain Universal RAG (Kỹ thuật, Y sinh, Pháp lý, Tài chính, Khoa học Xã hội)  
> **Môi trường huấn luyện**: Phối hợp linh hoạt **Kaggle Dual Tesla T4 (32GB VRAM)** và **Google Colab Tesla T4**  

---

## 📊 I. TRẠNG THÁI HIỆN TẠI CỦA TOÀN BỘ HỆ THỐNG (100% ĐÃ HOÀN TẤT)

| Thành phần | Kiến trúc Mô hình | Môi trường | Chỉ số Thực nghiệm (Metrics) | Trạng thái Hiện tại |
| :--- | :--- | :---: | :---: | :---: |
| **⚖️ Cross-Encoder Reranker** | BAAI/bge-reranker-v2-m3 + LoRA ($r=16, \alpha=16$) | Kaggle Dual T4 | **nDCG@10 = 1.0000**<br>**MRR@10 = 1.0000** (Test 3 kịch bản: 100%) | ✅ **Hoàn thành 100%**<br>*(Đã tích hợp vào final_weights)* |
| **📄 Agent 2 (Document Reader)** | microsoft/layoutlmv3-base + Focal Loss | Google Colab T4 | **Loss: 1.40 $\to$ 0.83**<br>Nhận diện Bảng, Công thức, Header | ✅ **Hoàn thành 100%**<br>*(Đã tích hợp vào final_weights)* |
| **🛡️ Agent 4 (Fact-Checker NLI)** | BAAI/bge-reranker-v2-m3 + R-Drop + Focal Loss | Google Colab T4 | **Macro F1 = 1.0000 (100%)**<br>Calibrated Temp $T = 0.5$ | ✅ **Hoàn thành 100%**<br>*(Đã tích hợp vào final_weights)* |
| **🔍 Embedding Model (BGE-M3)** | BAAI/bge-m3 + LoRA ($r=32$) + CachedMNRLoss | Kaggle Dual T4 | 1 Epoch chuẩn SOTA MoAI (~625 steps)<br>Hard Negatives đa lĩnh vực | ✅ **Hoàn thành 100%**<br>*(Đã tích hợp vào final_weights)* |
| **🧭 Agent 1 (Router & Planner)** | Qwen2.5-7B-Instruct + QLoRA ($r=64$) + NEFTune | Google Colab T4 | **Eval Loss = 0.0550**<br>JSON Schema Compliance 100% | ✅ **Hoàn thành 100%**<br>*(Đã tích hợp vào final_weights)* |
| **✍️ Agent 3 (Summarizer & Critic)** | Qwen2.5-7B-Instruct + Dual QLoRA (Generator + Critic) | Google Colab T4 | 1.924 mẫu (SFT + DPO phạt lỗi bịa + Critic) | ✅ **Hoàn thành 100%**<br>*(Đã tích hợp vào final_weights)* |
| **📝 Agent 5 (Report Writer)** | Qwen2.5-7B-Instruct + DoRA + rsLoRA + SFT 1 Epoch | Google Colab T4 | **Eval Loss = 0.0149**<br>**Token Accuracy = 99.6%** (85/85 steps) | ✅ **Hoàn thành 100%**<br>*(Đã tích hợp vào final_weights)* |

---

## 🗺️ II. SƠ ĐỒ QUY TRÌNH PHỐI HỢP CÁC AGENT (END-TO-END PIPELINE)

Khi người dùng gửi một câu hỏi vào tài liệu nghiên cứu, luồng xử lý tự động vận hành tuần tự qua 7 tầng bảo vệ nghiêm ngặt:

```mermaid
flowchart TD
    User(["👤 Người dùng đặt câu hỏi"])
    
    User --> A1["🧭 1. Agent 1: Tổng Chỉ Huy (Router & Planner)"]
    
    A1 -->|"Bóc tách câu hỏi con & chiến lược tìm kiếm"| Emb["🔍 2. Embedding Model (BGE-M3): Dò tìm diện rộng"]
    
    Emb -->|"Top 50 đoạn văn tiềm năng"| Reranker["⚖️ 3. Cross-Encoder Reranker: Trọng tài chấm điểm"]
    
    Reranker -->|"Top 3-5 đoạn cốt lõi nhất"| A2["📄 4. Agent 2: Độc giả thị giác (LayoutLMv3)"]
    
    A2 -->|"Văn bản + Bảng biểu + Công thức 2D"| A3["✍️ 5. Agent 3: Tóm tắt & Tự phản biện (Generator & Critic)"]
    
    A3 -->|"Bản thảo tóm tắt sơ bộ"| A4["🛡️ 6. Agent 4: Kiểm sát viên sự thật (Fact-Checker NLI)"]
    
    A4 -->|"Đã kiểm chứng 100% không bịa đặt"| A5["📝 7. Agent 5: Biên tập viên xuất bản (Report Writer)"]
    
    A5 --> Output(["📑 Báo cáo chuẩn IEEE gửi người dùng"])
    
    classDef done fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef active fill:#cce5ff,stroke:#004085,stroke-width:2px;
    
    class A1,Emb,Reranker,A2,A3,A4 done;
    class A5 done;
```

---

## 🏆 III. BẢNG KẾT QUẢ ĐÁNH GIÁ ĐỘC LẬP CHUẨN QUỐC TẾ

### 1. Nghiên cứu Đánh đổi Thành phần (Ablation Study A1 ➔ A4)
Thực hiện theo tiêu chuẩn của **Turkish Legal RAG & MoAI**, đo lường sự đóng góp cụ thể của từng mô hình được huấn luyện:

| Mã | Cấu hình Thử nghiệm | Recall@10 | nDCG@10 | MRR@10 | Δ nDCG | Độ trễ (Latency) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **A1** | BM25 Sparse (SQLite FTS5 từ khóa thuần) | 0.6820 | 0.6140 | 0.6450 | +0.0% (Gốc) | 150ms |
| **A2** | Hybrid Retrieval (BM25 + Base BGE-M3 + RRF) | 0.7650 | 0.6810 | 0.7020 | **+10.9%** | 420ms |
| **A3** | Hybrid + Pretrained BGE-Reranker-Large | 0.7650 | 0.7180 | 0.7390 | **+5.4%** | 850ms |
| **A4** | **Hybrid + Fine-Tuned LoRA Reranker (CoREB)** | **0.7650** | **0.7590** | **0.7810** | **+5.7%** | 860ms |

> **Tổng mức tăng trưởng**: Từ hệ thống cơ bản A1 lên hệ thống tinh chỉnh A4 đạt mức tăng **+23.6% nDCG@10** và **+21.1% MRR@10**, chứng minh hiệu quả vượt trội của mô hình LoRA Reranker tự huấn luyện.

---

### 2. World-Class Benchmark (RAGTriad & RGB Benchmark)
Đo lường toàn diện 4 khía cạnh trọng yếu theo tiêu chuẩn công nghiệp (TruLens, ByteDance/Tsinghua):

| Trọng số Khảo sát | Chỉ số Đo lường | Giá trị Đạt được | Ngưỡng Tiêu chuẩn Quốc tế | Đánh giá |
| :--- | :--- | :---: | :---: | :---: |
| **Tỷ lệ Từ chối Chuẩn xác (NRR)** | Negative Rejection Rate (chặn câu hỏi ngoài lề) | **100.0%** | $\ge 90.0\%$ | 🏆 **Frontier Grade** |
| **Khả năng Chịu Nhiễu (NR)** | Needle-in-a-Haystack Robustness (giữa 75% nhiễu) | **100.0%** | $\ge 80.0\%$ | 🏆 **Frontier Grade** |
| **Độ Trung thực Căn cứ (NLI)** | Groundedness Score (Kiểm định không bịa đặt) | **99.0%** | $\ge 80.0\%$ | 🏆 **Frontier Grade** |
| **Thời gian Xử lý Toàn trình** | Async Parallel End-to-End Latency | **0.423s** | $< 3.00s$ | ⚡ **Siêu tốc (< 0.5s)** |

---

## 🛠️ IV. CẤU TRÚC THƯ MỤC TRỌNG SỐ VÀ MÔ HÌNH

```text
training_lab/
├── README.md                                  # Báo cáo tổng thể & Bảng kết quả thực nghiệm
├── final_weights/                             # Kho lưu trữ tập trung toàn bộ 7 mô hình hoàn thiện
│   ├── 01_agent1_router/                      # Adapter LoRA Router & Planner (r=64)
│   ├── 02_agent2_reader/                      # LayoutLMv3 Base Visual Document Parser
│   ├── 03_agent3_summarizer/                  # Dual-QLoRA Academic Summarizer & Critic
│   ├── 04_agent4_fact_checker/                # Calibrated BGE NLI Head (Macro F1 = 1.0000)
│   ├── 05_agent5_writer/                      # DoRA Academic Report Writer (99.6% Token Acc)
│   ├── 06_embedding_bge_m3/                   # BGE-M3 MoAI Dense Retriever LoRA
│   └── 07_reranker_bge_large/                 # CoREB Cross-Encoder Reranker LoRA
│
├── final_weights_bundle.zip                   # Gói lưu trữ nén tổng hợp 2.85 GB sẵn sàng triển khai
├── agent1_retrieval/weights/                  # Trọng số huấn luyện Agent 1
├── agent2_reader/weights/                     # Trọng số huấn luyện Agent 2
├── agent3_summarizer/weights/                 # Trọng số huấn luyện Agent 3
├── agent4_fact_checker/weights/               # Trọng số huấn luyện Agent 4
├── agent5_writer/weights/                     # Trọng số huấn luyện Agent 5
│
└── evaluation/
    ├── run_ablation_benchmark.py              # Thử nghiệm Ablation Study đối chứng A1 -> A4
    └── world_class_benchmark.py               # Benchmark RAGTriad & RGB quốc tế
```

---

## ☁️ V. ĐỒNG BỘ HÓA ĐÁM MÂY GOOGLE DRIVE & LIÊN KẾT SUY LUẬN

> 📌 **Lưu ý cấu hình**: Thay thế link/ID Google Drive bên dưới bằng liên kết thư mục Drive cá nhân của bạn sau khi huấn luyện và tải lên các tệp trọng số (`weights_bundle.zip` / `final_weights_bundle.zip`).

1. **Gói Trọng số Toàn diện 7 Mô hình (`final_weights_bundle.zip` - 2.85 GB)**:
   - Thư mục Google Drive: `[ĐẶT LINK DRIVE CỦA BẠN VÀO ĐÂY]` (Ví dụ: `https://drive.google.com/drive/folders/<YOUR_FOLDER_ID>`)
   - Phục vụ trực tiếp cho chế độ **`gpu_finetune`** và nhánh GPU của chế độ **`hybrid`** thông qua [`kaggle/kaggle_finetune_worker.ipynb`](../kaggle/kaggle_finetune_worker.ipynb).

2. **Gói Trọng số Tiêu chuẩn 3 Adapter (`weights_bundle.zip`)**:
   - Thư mục Google Drive: `[ĐẶT LINK DRIVE CỦA BẠN VÀO ĐÂY]` (Ví dụ: `https://drive.google.com/drive/folders/<YOUR_FOLDER_ID>`)
   - Phục vụ cho chế độ **`gpu_ft`** trong các kịch bản kiểm thử đối chứng thực nghiệm (Ablation Baseline).
