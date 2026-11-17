# 🛡️ Agent 4 (`agent4_fact_checker`) — 3-Class Cross-Encoder NLI Verifier (Kaggle T4x2)

## Kiến trúc thư mục `agent4_fact_checker/`
```text
agent4_fact_checker/
├── src/
│   ├── __init__.py          # Package interface
│   ├── config.py            # Agent4TrainingConfig & ánh xạ 3 nhãn (SUPPORT, CONTRADICT, NEI)
│   ├── data_loader.py       # Đọc nli_verifier.jsonl và mã hóa cặp (claim, evidence)
│   ├── model_builder.py     # Khởi tạo BAAI/bge-reranker-v2-m3 / SciBERT 3-class
│   ├── trainer.py           # WeightedCrossEncoderTrainer (nhân hệ số phạt 1.35x cho CONTRADICT)
│   └── evaluator.py         # Tính Macro-F1 & hiệu chỉnh nhiệt độ xác suất (Temperature ECE)
├── dataset/
│   └── nli_verifier.jsonl   # 3,000 cặp thẩm định sự kiện khoa học
├── weights/                 # Trọng số Cross-Encoder NLI & eval_metrics.json
├── train_agent4_t4x2.py     # CLI Entrypoint
└── Train_Agent4_FactChecker_T4x2.ipynb
```
