# Bật Cohere Rerank API

Backend đã hỗ trợ `POST https://api.cohere.com/v2/rerank` qua httpx sẵn có,
không cần cài thêm SDK. Tham chiếu: https://docs.cohere.com/v2/reference/rerank.

## Cấu hình

Sửa các dòng tương ứng trong `.env` ở thư mục gốc (không tạo dòng trùng):

```dotenv
RA_RERANKER_PROVIDER=cohere
RA_RERANKER_MODEL=rerank-v4.0-pro
RA_COHERE_API_KEY=dan_key_trial_cua_ban_vao_day
RA_COHERE_BASE_URL=https://api.cohere.com/v2
RA_RERANK_CANDIDATES=40
RA_RERANKER_REQUESTS_PER_MINUTE=8
RA_RERANKER_TIMEOUT_SECONDS=60
RA_RERANKER_CACHE_TTL_SECONDS=86400
RA_RERANKER_MAX_TOKENS_PER_DOC=4096
RA_RERANKER_FAILURE_POLICY=fallback
```

Nhập key trực tiếp vào tệp trên máy, không dán key vào chat hoặc commit `.env`.
Khởi động lại backend. Không cần upload lại PDF, rebuild hoặc tạo lại embedding.
Nếu model không được tài khoản cho phép, backend sẽ thông báo lỗi quyền truy cập;
việc có Trial key không tự xác nhận mọi model đều gọi thành công.

`auto` giữ tương thích: có `RA_RERANKER_MODEL` thì dùng local, để trống thì tắt.
Muốn dùng Cohere cần chọn rõ `cohere`. `none` luôn tắt reranker; `local` dùng model local.

## Xác nhận đang hoạt động

- `/api/config`: `reranker_provider` phải là `cohere`, model đúng tên đã cấu hình.
- `/api/diagnostics`: `reranker.api` hiển thị số lượt API, cache hit và lỗi từ khi backend
  khởi động. Các trường này không chứa key.
- Trace của lượt nghiên cứu: `retrieval_lanes[].reranker` có provider/model,
  `cache_hit`, `api_calls` và `fallback`. `after_rerank` chứa điểm cho từng chunk.
- Bộ đếm gọi LLM/token hiện có dành cho các agent ngôn ngữ. Lượt Cohere rerank được
  ghi riêng trong trace/diagnostics, không cộng vào số token LLM hoặc ước tính phí LLM.

## Tiết kiệm quota và xử lý lỗi

Mỗi truy vấn xếp hạng nhiều đoạn trong một request. Ví dụ 40 đoạn không có nghĩa là
40 request. Nhiều ý hoặc hai vòng nghiên cứu có thể tạo nhiều truy vấn/request.

Cache dùng endpoint, model, câu hỏi, toàn bộ nội dung đoạn theo thứ tự và giới hạn token
để tạo khóa. Đổi văn bản sẽ tạo khóa mới; điểm cache luôn gắn lại vào metadata hiện tại.
Các yêu cầu trùng đồng thời dùng chung kết quả. `RA_CACHE_ENABLED=false` hoặc
`use_cache=false` của lượt chạy sẽ bỏ qua cache; `/api/cache/clear` xóa cả cache rerank.

Giới hạn 8 request/phút được dùng chung cho các phiên trong một tiến trình backend.
Nó không điều phối với chương trình khác cùng dùng key. Hạn mức tháng do Cohere quản lý;
xem Billing & Usage để biết số lượt còn lại. Tài liệu hạn mức:
https://docs.cohere.com/v2/docs/rate-limits.

Lỗi 429 không tự gọi lại. Backend tôn trọng `Retry-After` (mặc định 60 giây nếu thiếu),
trong thời gian đó lượt chưa có cache sẽ báo không khả dụng ngay. Điều này tránh giữ
phiên chờ lâu khi hết quota tháng. Không tự chuyển sang key hoặc nhà cung cấp khác.

- `fallback`: tiếp tục bằng thứ tự Hybrid/BM25 và hiển thị cảnh báo cùng nguyên nhân.
  Đây không phải fallback sang model local; muốn dùng local cần cấu hình riêng.
- `error`: dừng lượt nghiên cứu khi reranker lỗi. Có thể chọn chế độ này nếu muốn bảo
  đảm mọi lượt thành công đều đã đi qua Cohere.

Không tự retry lỗi xác thực, lỗi model hoặc phản hồi không hợp lệ. Chỉ số sai, trùng,
thiếu kết quả, điểm không hợp lệ đều bị loại trước khi truyền tới Reader.

## Kiểm thử

`python -m pytest tests/test_cohere_reranker.py -q` dùng HTTP giả lập, không tiêu quota.
Sau khi thêm key và khởi động lại, chạy câu hỏi trong giao diện để kiểm tra tài khoản
thực tế. Trial key được dùng miễn phí trong hạn mức và điều kiện của Cohere; tích hợp
không kiểm tra hoặc tự nâng cấp gói thanh toán.
