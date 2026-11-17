# 📝 Agent 5 (`agent5_writer`) — Academic Report Writer & Citation Synthesizer (Kaggle T4x2)

## Kiến trúc SOTA 2025/2026 của `agent5_writer/`
- **Backbone**: `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 + `attn_implementation="sdpa"` tối ưu cho Turing `sm_75` Tesla T4).
- **PEFT**: **DoRA (`use_dora=True`) + rsLoRA (`use_rslora=True`)** giúp ổn định đạo hàm theo $\frac{\alpha}{\sqrt{r}}$ và tách rời độ lớn/hướng vector trọng số.
- **Huấn luyện 2 giai đoạn (SFT + RPO/DPO)**:
  1. **Stage 1 (SFT + NEFTune $\alpha=5.0$)**: Học cấu trúc báo cáo khoa học chuẩn IEEE/Markdown (Executive Summary, Phân tích Kỹ thuật + LaTeX, Bảng đối chiếu Thực nghiệm, Danh mục Trích dẫn `[DocX]`).
  2. **Stage 2 (RPO/DPO Alignment với `rpo_alpha=0.1`)**: Tối ưu hóa tỷ lệ ưu tiên kèm thành phần phạt NLL giúp mô hình không bao giờ bỏ sót trích dẫn `[DocX]` đã được Agent 4 thẩm định.

```text
agent5_writer/
├── src/
│   ├── __init__.py
│   ├── config.py            # Agent5TrainingConfig (DoRA + rsLoRA + SDPA + RPO/DPO)
│   ├── data_loader.py       # Nạp writer_sft.jsonl (1,500) & writer_dpo.jsonl (1,000)
│   ├── model_builder.py     # Qwen2.5-7B 4-bit NF4 cân bằng tải trên Dual T4
│   ├── trainer.py           # Two-Stage SFT + RPO/DPO Trainer & Xuất FP16 (<80MB)
│   └── evaluator.py         # Kiểm định cấu trúc Markdown, Bảng, LaTeX & Citation [DocX]
├── dataset/
│   ├── writer_sft.jsonl     # 1,500 mẫu tổng hợp báo cáo khoa học
│   └── writer_dpo.jsonl     # 1,000 cặp đối chiếu ưu tiên (Chosen vs Rejected)
├── weights/                 # Trọng số QLoRA FP16 (19.28 MB - nạp trực tiếp làm Adapter thứ 4 trên Qwen2.5-7B)
├── train_agent5_t4x2.py     # CLI Entrypoint
└── Train_Agent5_Writer_T4x2.ipynb
```
