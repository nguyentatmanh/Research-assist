# Chạy Research Assist Master Worker trên Kaggle

Thêm `dist/ResearchDesk_Source.zip` vào một Kaggle Dataset riêng tư rồi gắn dataset vào [kaggle_master_worker.ipynb](kaggle_master_worker.ipynb). Notebook kiểm tra SHA-256 của mã nguồn và giải nén vào thư mục riêng dưới `/kaggle/temp`. Tạo `RA_ACCESS_TOKEN` trong Kaggle Secrets và cấp quyền cho notebook. Token gồm 32–256 ký tự chữ, số, dấu gạch dưới hoặc gạch ngang.

Chọn `workspace` để dùng web, `gpu_worker` để dùng dịch vụ suy luận từ máy khác, hoặc `hybrid` để chạy cả hai. Chọn GPU T4 ×2 hoặc GPU đơn nếu chạy worker. Bật Internet để tải thư viện và trọng số. Đặt `OPEN_PUBLIC_TUNNEL = True` trong notebook khi muốn mở kết nối từ ngoài Kaggle. Gateway ở cổng 8000 và các tiến trình con chỉ lắng nghe trên loopback. Launcher kiểm tra dịch vụ và xác thực trước khi in URL. Dùng token lấy từ Kaggle Secrets để đăng nhập.

Nếu đã có cây mã nguồn từ ZIP, chạy `bash kaggle/boot.sh --mode hybrid` tại thư mục dự án. Thêm `--public-tunnel` để mở kết nối ngoài Kaggle. Đặt `RA_GEMINI_API_KEY` trong Secrets nếu dùng Gemini cho Workspace và `RA_OPENALEX_API_KEY` nếu cần hạn mức OpenAlex theo tài khoản. Dữ liệu và nhật ký mặc định nằm trong `/kaggle/temp/research-assist-master`.

## Kết nối GPU

Các route trong bảng dưới đây yêu cầu header `Authorization: Bearer <RA_ACCESS_TOKEN>`. `GET /health` báo trạng thái sẵn sàng ở chế độ hiện tại.

| Phương thức | Đường dẫn | Chức năng |
|---|---|---|
| GET | `/v1/info` | Nạp embedding E5 và trả model ID, tokenizer, giới hạn token và chiều vector |
| POST | `/v1/embeddings` | Nhận `model`, `input` và `input_type` (`query` hoặc `passage`) |
| POST | `/api/embed` | Cũng hỗ trợ định dạng cũ với `texts` và `is_query` |
| POST | `/api/rerank` | Xếp hạng `documents` theo `query` |
| GET | `/v1/models` | Danh sách LLM đã nạp. Master Worker mặc định nạp embedding và reranker |
| POST | `/api/admin/switch_mode` | Đổi chế độ với JSON `{"mode":"hybrid"}` |
| GET | `/api/admin/logs?service=workspace&lines=50` | Đọc tối đa 200 dòng cuối từ nhật ký `workspace`, `gpu_worker` hoặc `supervisor` |
| GET | `/api/admin/vram` | Đọc bộ nhớ trống trên thiết bị, bộ nhớ PyTorch đã cấp phát, đã dự trữ và số sự cố OOM |
| POST | `/api/admin/vram/clear` | Giải phóng cache CUDA đang rảnh nếu worker không bận suy luận |
| POST | `/api/admin/worker/reset` | Khởi động lại GPU Worker đã bật sau khi xác nhận worker rảnh |

Trong trang cấu hình GPU Embedding của Research Assist trên máy local, điền URL của gateway, token và model ID `research-e5-…` từ `/v1/info`. Dùng tokenizer và giới hạn token trả về từ cùng endpoint. Giữ model ID cũ khi chỉ đổi địa chỉ tunnel. Lập lại chỉ mục khi đổi checkpoint. Dùng notebook `06_kaggle_llm_worker.ipynb` cho nhu cầu chạy LLM trên GPU.

Với hai GPU, worker đặt Embedding và Reranker trên GPU 1. Trên một GPU, worker dùng GPU 0. Dọn cache giữ mô hình đang nạp và trả số đo trước, sau. Reset giải phóng tiến trình để nạp lại mô hình ở lượt gọi tiếp theo. Cả hai thao tác trả mã 409 nếu worker đang suy luận. Dùng các số đo này và độ trễ suy luận để đánh giá hiệu quả trên Dual T4 thật.
