# Chạy Research Assist trên Kaggle

Chọn notebook theo nơi bạn muốn chạy giao diện. Notebook **04** chạy cả giao diện
và backend trên Kaggle. Notebook **03** và **06** chạy dịch vụ GPU để Research Assist
trên máy bạn kết nối. Có thể thử với model gốc trước khi huấn luyện.

| Notebook | Dùng khi |
| --- | --- |
| [04_kaggle_workspace.ipynb](../notebooks/04_kaggle_workspace.ipynb) | Muốn mở toàn bộ Research Assist từ trình duyệt. Mặc định không cần GPU hay khóa nhà cung cấp mô hình. |
| [03_gpu_server.ipynb](../notebooks/03_gpu_server.ipynb) | Muốn Kaggle tính embedding. Mặc định dùng Qwen3 Embedding 0.6B, có thể nạp adapter 01 hoặc checkpoint E5. |
| [06_kaggle_llm_worker.ipynb](../notebooks/06_kaggle_llm_worker.ipynb) | Muốn các agent dùng LLM chạy trên GPU Kaggle. Có Qwen2.5 7B AWQ trên một GPU và profile thử nghiệm 32B AWQ tensor parallel trên hai GPU. |
| [kaggle_master_worker.ipynb](../kaggle/kaggle_master_worker.ipynb) | Muốn dùng gateway chung cho Workspace và GPU Worker. Xem `kaggle/README.md` trong Source ZIP để cấu hình token, chế độ chạy và thao tác bảo trì GPU. |
| [01_train_embedding.ipynb](../notebooks/01_train_embedding.ipynb) | Muốn huấn luyện adapter Qwen3 Embedding. Đầu ra `qwen3-adapter.zip` có thể nạp vào 03. |
| [02_train_reranker.ipynb](../notebooks/02_train_reranker.ipynb) | Muốn thử nghiệm huấn luyện reranker Qwen3. Đầu ra `qwen3-reranker-adapter.zip` kèm mã chấm điểm. Đây là artifact thử nghiệm riêng, chưa có ô nạp trực tiếp trong UI. |

## Chạy cả giao diện và backend với notebook 04

1. Trong PowerShell tại thư mục dự án, chạy:

   ```powershell
   .\.venv\Scripts\python.exe scripts/package_source.py
   .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. Tải `dist/ResearchDesk_Source.zip` lên một Kaggle Dataset riêng tư. Import notebook
   04, chọn **Add Input** để gắn Dataset đó, bật **Internet**. Nếu Kaggle gắn nhiều ZIP
   cùng tên, điền đường dẫn đúng vào `SOURCE_ZIP`. Notebook gọi [boot.sh](../notebooks/boot.sh)
   từ mã nguồn đã giải nén, nên chỉ cần gắn ZIP và import notebook.
3. Trong **Add-ons → Secrets**, thêm `RA_ACCESS_TOKEN` bằng giá trị vừa tạo và cấp
   quyền cho notebook. Nếu muốn dùng Gemini, thêm `RA_GEMINI_API_KEY` và đặt
   `USE_GEMINI = True`. Nếu muốn dùng GPU cho truy xuất, bật GPU trong Kaggle và đặt
   `USE_GPU_RETRIEVAL = True`. Literature Scout hỗ trợ tra cứu OpenAlex cơ bản không
   cần key. Nếu muốn dùng hạn mức theo tài khoản OpenAlex, thêm Secret
   `RA_OPENALEX_API_KEY` và đặt `USE_OPENALEX_KEY = True`.
4. Đặt `START_WORKSPACE = True`. Để mở từ trình duyệt bên ngoài Kaggle, đặt thêm
   `OPEN_PUBLIC_TUNNEL = True` nếu môi trường cho phép, rồi chạy lần lượt từ đầu
   đến cell khởi động. Mở URL HTTPS được in ra và đăng nhập bằng `RA_ACCESS_TOKEN`.
   Notebook kiểm tra health, API từ chối khách chưa xác thực và API nhận đúng token
   trước khi in URL. Khi lỗi khởi động, xem đường dẫn `workspace.log` trong output cell.
5. Nạp một tài liệu mẫu hoặc tải PDF, chọn tài liệu và đặt câu hỏi. Chế độ Demo
   dùng trích xuất để thử luồng; chọn Gemini hoặc cấu hình model khác để dùng LLM.
   Khi xong, đặt `STOP_WORKSPACE = True` trong cell cuối và chạy riêng cell đó.

Tài liệu, SQLite, lịch sử và cấu hình nằm trong `/kaggle/temp/research-assist-data`.
Chúng tồn tại khi khởi động lại app trong cùng phiên, và mất khi Kaggle hủy phiên.
Tải báo cáo và tài liệu cần lưu trước khi đóng. URL Quick Tunnel cũng đổi khi mở
lại tunnel. Dùng phiên tương tác để chạy app/worker. [Kaggle Notebooks][kaggle] ·
[Cloudflare Quick Tunnels][cloudflare].

Nếu đã cài ứng dụng và nạp `RA_ACCESS_TOKEN` vào môi trường, có thể chạy riêng launcher
bằng `bash notebooks/boot.sh /kaggle/temp/research-assist-runtime/venv/bin/python 8000`.
Launcher mở backend trên loopback. Notebook quản lý Quick Tunnel và kiểm tra đăng nhập.

## Giữ giao diện trên máy và dùng GPU Kaggle

**Embedding (03).** Bật Internet và GPU, tạo Secret `RESEARCH_GPU_API_KEY` bằng
token ngẫu nhiên ít nhất 32 ký tự. Đặt `START_GPU_SERVER = True` và, nếu kết nối từ
ngoài Kaggle, `START_CLOUDFLARE = True`. Chạy từ đầu. Trong Research Assist, mở cấu
hình **GPU embedding** và nhập URL, token, model ID và tokenizer theo output.
Kiểm tra kết nối rồi lưu. Khi chuyển sang checkpoint/model khác, dùng **Lập lại
chỉ mục** trước khi dùng kho tài liệu với model mới. Cùng model ID, đổi URL tunnel
thì giữ được chỉ mục.

**LLM (06).** Bật Internet và GPU, gắn Secret `RESEARCH_GPU_API_KEY`, chọn
`MODEL_PROFILE = "7b"` cho lần đầu. Đặt `START_LLM_WORKER = True` và
`OPEN_PUBLIC_TUNNEL = True` khi cần kết nối bên ngoài. Sau khi smoke test JSON
schema thành công, sao chép URL `/v1` và model ID. Trong Research Assist, mở cấu
hình **GPU Worker**, chọn provider OpenAI, nhập URL, model, token và kiểm tra kết
nối. Chọn GPU cho agent muốn chuyển sang model này. Profile 32B cần hai GPU và
có thể không nạp được tùy VRAM và CUDA driver của phiên. Nếu hết bộ nhớ, quay lại
7B. GPU embedding và GPU Worker có cấu hình riêng.

## Huấn luyện trên Kaggle

Notebook 01 và 02 tự chuẩn bị QASPER mặc định hoặc nhận dữ liệu đã chuẩn bị từ
Kaggle Input. Bật GPU và Internet, chỉnh cell cấu hình trước khi chạy.
Checkpoint nằm dưới `/kaggle/working/<RUN>/`; lưu phiên bản có Output và tải
ZIP, hoặc gắn Output đó làm Input cho phiên khác. Với 01, dùng đường dẫn
`qwen3-adapter.zip` ở `CHECKPOINT_ZIP` của 03. QASPER mặc định đo truy xuất trên
dữ liệu tiếng Anh. Kết quả đó không đo chất lượng trả lời tiếng Việt.

## Luồng được tham khảo

[Notebook master/worker][reference] của Multi-Agent_RAG tải mã nguồn, phát hiện
GPU, chạy gateway rồi in URL. Tại bản đối chiếu, thư mục chỉ còn notebook đó.
[Bản cũ][archived] có bốn notebook cho triển khai từ xa, worker cơ bản,
worker fine-tune và vLLM 32B. Research Assist dùng chính hợp đồng API và checkpoint
của mình, với hai notebook worker độc lập và một notebook riêng để chạy cả web.

[kaggle]: https://www.kaggle.com/docs/notebooks
[cloudflare]: https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/
[reference]: https://github.com/ducmanh-jr/Multi-Agent_RAG/blob/ebbb38654acec9aff06fe1c143e66362a2b60eac/kaggle/kaggle_master_worker.ipynb
[archived]: https://github.com/ducmanh-jr/Multi-Agent_RAG/tree/a3e3b3083725c94244e4a65337b648125626441d/kaggle
