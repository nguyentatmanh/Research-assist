# Embedding và reranker

Agent Retrieval có lớp lập kế hoạch, embedding và reranking. Adapter lập kế hoạch được huấn luyện trong [agent1_retrieval](../agent1_retrieval). Dùng các luồng Research-assist hiện có cho hai lớp còn lại.

| Thành phần | Huấn luyện | Nạp vào ứng dụng |
|---|---|---|
| E5 embedding | [CLI train](../../scripts/train_retriever.py) hoặc [builder notebook E5](../../scripts/build_kaggle_notebooks.py) | `RA_EMBEDDING_PROVIDER=sentence-transformers`, `RA_EMBEDDING_MODEL` và `RA_EMBEDDING_FAMILY=e5`. Tạo lại chỉ mục embedding sau khi đổi model. |
| Qwen3 embedding | [Notebook huấn luyện](../../notebooks/01_train_embedding.ipynb) và [notebook worker GPU](../../notebooks/05_gpu_qwen3_inference_cloudflare.ipynb) | Dùng endpoint worker embedding hoặc checkpoint tương thích với [cấu hình embedding](../../src/research_assist/config.py). Tạo lại chỉ mục sau khi đổi model. |
| Qwen3 reranker | [Notebook reranker](../../notebooks/02_train_reranker.ipynb) và [builder Qwen3](../../scripts/build_qwen_reranker_notebook.py) | Định dạng checkpoint hoàn chỉnh có thể nạp qua local reranker tương thích. Với adapter Qwen3, thực hiện quy trình export và đánh giá của notebook trước khi triển khai. |

So sánh BM25, hybrid và reranking trên cùng tập test bằng [CLI đánh giá truy xuất](../../scripts/evaluate_retrieval.py) và các biến thể trong [`configs`](../../configs).
