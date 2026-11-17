# 📄 Agent 2 (`agent2_reader`) — Document Layout & Formula/Table Parser (Kaggle T4x2)

## Kiến trúc thư mục `agent2_reader/`
```text
agent2_reader/
├── src/
│   ├── __init__.py          # Package interface
│   ├── config.py            # Agent2TrainingConfig & 11 nhãn cấu trúc DocLayNet
│   ├── data_loader.py       # Bộ nạp tokens + tọa độ không gian 2D bboxes + ner_tags
│   ├── model_builder.py     # Khởi tạo microsoft/layoutlmv3-base (11 classes)
│   ├── trainer.py           # Pipeline huấn luyện & xuất weights + label_map.json
│   └── evaluator.py         # Kiểm định tập nhãn phân đoạn tài liệu khoa học
├── dataset/
│   └── layout_doclaynet_sample.jsonl
├── weights/                 # Trọng số LayoutLMv3 đã tinh chỉnh
├── train_agent2_t4x2.py     # CLI Entrypoint
└── Train_Agent2_Reader_T4x2.ipynb
```
