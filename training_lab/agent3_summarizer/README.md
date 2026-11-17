# ✍️ Agent 3 (`agent3_summarizer`) — Generator (SFT+DPO) & Critic (Kaggle T4x2)

## Kiến trúc thư mục `agent3_summarizer/`
```text
agent3_summarizer/
├── src/
│   ├── __init__.py          # Package interface
│   ├── config.py            # Agent3TrainingConfig (Generator r=64 SFT+DPO & Critic r=32)
│   ├── data_loader.py       # Nạp generator_sft.jsonl, generator_dpo.jsonl, critic_sft.jsonl
│   ├── model_builder.py     # Qwen2.5-7B-Instruct 4-bit NF4 cân bằng tải trên 2 GPU T4
│   ├── trainer.py           # Huấn luyện 2 giai đoạn (Stage 1 SFT -> Stage 2 DPO) & Critic SFT
│   └── evaluator.py         # Kiểm định trọng số FP16 (<80MB) cho cả Generator và Critic
├── dataset/
│   ├── generator_sft.jsonl  # 1,500 mẫu SFT tóm tắt kèm trích dẫn
│   ├── generator_dpo.jsonl  # 1,000 cặp DPO (Chosen vs Rejected)
│   └── critic_sft.jsonl     # 1,500 mẫu SFT phản biện bắt lỗi ảo giác
├── weights/
│   ├── generator_qlora/     # Trọng số Generator FP16 (77.05 MB)
│   └── critic_qlora/        # Trọng số Critic FP16 (19.28 MB)
├── train_agent3_t4x2.py     # CLI Entrypoint
└── Train_Agent3_Summarizer_T4x2.ipynb
```
