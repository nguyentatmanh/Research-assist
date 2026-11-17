# 🧭 Agent 1 (`agent1_retrieval`) — Retrieval Router & Query Planner (Kaggle T4x2)

## Kiến trúc thư mục `agent1_retrieval/`
```text
agent1_retrieval/
├── src/
│   ├── __init__.py          # Package interface
│   ├── config.py            # Agent1TrainingConfig (DoRA r=64, alpha=128, Dual T4 14.5GiB/GPU)
│   ├── data_loader.py       # Đọc router_sft.jsonl, kiểm tra JSON schema, chia 90/10 Train/Val
│   ├── model_builder.py     # Qwen2.5-7B-Instruct 4-bit NF4 + device_map="balanced" + DoRA
│   ├── trainer.py           # SFTTrainer (NEFTune=5.0, Paged AdamW 8-bit) + Ép kiểu FP16 (<80MB)
│   └── evaluator.py         # Đánh giá tỷ lệ tuân thủ JSON Schema sau huấn luyện
├── dataset/
│   └── router_sft.jsonl     # 1,500 mẫu ChatML điều phối & phân rã truy vấn
├── weights/                 # Trọng số QLoRA FP16 (77.05 MB)
├── train_agent1_t4x2.py     # CLI Entrypoint
└── Train_Agent1_Router_T4x2.ipynb
```
