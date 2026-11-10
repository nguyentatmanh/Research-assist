# Nâng cấp nghiên cứu tài liệu — nhánh `upgrade`

## Phạm vi đã tích hợp

Các ý tưởng được tham khảo từ [ducmanh-jr/Multi-Agent_RAG](https://github.com/ducmanh-jr/Multi-Agent_RAG/tree/35ab1b50b02b1a61dba726032aa3aeced531bcdc), sau đó triển khai độc lập theo kiến trúc hiện có. Không sao chép mã nguồn, điểm đánh giá hoặc kết quả benchmark của repo tham khảo.

| Ưu tiên | Thay đổi | Cách kiểm chứng |
| --- | --- | --- |
| P0 | Bộ pilot 20 câu, chia đều `single`, `multi`, `table`, `unanswerable`; đánh dấu trang PDF vật lý, hàng/cột/giá trị tham chiếu cho câu hỏi bảng | Kiểm tra đoạn tham chiếu thực sự xuất hiện ở trang được gán; lưu hash tài liệu, cấu hình và trace |
| P1 | Nhận diện bảng có đường kẻ ngang và chú thích; giữ vị trí cột, ô trống, số mũ; gắn vùng nguồn loại `table` | PDF tổng hợp, PDF bài báo thật, đối chiếu tọa độ và trích dẫn |
| P1 | Reader tự chủ có công cụ `read_tables`, chỉ mở các bảng trên trang thuộc đoạn nguồn đã quan sát | Kiểm thử toàn luồng agent, giới hạn phạm vi và snapshot bất biến khi tài liệu gốc bị xóa |
| P1 | Fact Checker kiểm tra điều kiện thí nghiệm và ý nghĩa thống kê trong bằng chứng của từng nhận định | Không chấp nhận chênh lệch điểm hoặc câu phủ định như bằng chứng khẳng định ý nghĩa thống kê |
| P2 | Tìm trên arXiv, chọn phiên bản cụ thể, nhập PDF vào pipeline hiện có | Kiểm thử API, chống nhập trùng theo SHA-256, lưu `arxiv_id`, kiểm thử trình duyệt và gọi arXiv thật |

Giữ các cơ chế BM25/hybrid, reranker, bằng chứng có tọa độ, ngân sách agent và kiểm chứng số/ngày đã có. Không ưu tiên một bảng chỉ vì nó nằm cùng trang với kết quả tìm kiếm. Reader có thể mở bảng dựa trên nhu cầu của câu hỏi. Chế độ fixed và luồng song song vẫn dùng được văn bản bảng đã cải thiện qua truy xuất thông thường.

## Sử dụng sau khi cập nhật

1. Khởi động lại backend để nạp mã mới.
2. Với tài liệu đã nhập trước bản này, vào **Kho tài liệu → Quản lý chỉ mục → Chia lại đoạn & tạo lại chỉ mục**. Thao tác này cần để bổ sung cấu trúc bảng. **Reindex embedding** riêng lẻ không trích xuất lại PDF. Tài liệu nhập mới dùng cấu trúc mới ngay.
3. Trong Kho tài liệu, mở **Tìm bài liên quan trên arXiv**, nhập tiêu đề hoặc từ khóa, rồi chọn **Thêm vào kho** ở bài cần dùng. Tìm kiếm metadata không tự nhập PDF. Bài đã nhập không tự thay đổi tập nguồn đang chọn cho nghiên cứu.
4. Chọn tài liệu cần nghiên cứu và hỏi như trước. Với câu hỏi bảng, nêu tên bảng, chỉ số hoặc nhóm cần so sánh để giảm nhập nhằng.

Không thay đổi `.env`, API key, model/embedding đang cấu hình hoặc tự xây lại kho dữ liệu của người dùng. arXiv sử dụng client HTTP hiện có, tối đa 5 kết quả mỗi lần tìm và khoảng nghỉ tối thiểu 3 giây. PDF bị giới hạn dung lượng theo cấu hình upload; chỉ tải từ endpoint arXiv cố định, không đi theo chuyển hướng sang địa chỉ khác. Lỗi mạng và giới hạn truy cập được hiển thị để có thể thử lại.

`SourceRegion.kind` có giá trị mặc định `text`, `Document.arxiv_id` mặc định `null`, nên dữ liệu cũ vẫn đọc được. Phiên bản xử lý là `layout-section-tables-v3`; chính sách bằng chứng là `evidence-first-five-agents-v9`.

## Bộ pilot và khả năng tái lập

File: `sample_papers/upgrade_pilot.jsonl`. Đây là **dữ liệu phát triển do AI soạn, chưa được chuyên gia duyệt** (`label_source: unreviewed`), không phải gold test độc lập. Trường `expected_value` mô tả đáp án tham chiếu, không tự biến retrieval recall thành điểm đúng của câu trả lời. Các chỉ số chấm bởi con người giữ `null` cho đến khi có đánh giá thật.

Đặt hai PDF dưới đây cạnh file JSONL. Không kèm bản sao PDF trong commit nâng cấp:

| Tên file | Nguồn | SHA-256 của bản đã kiểm tra |
| --- | --- | --- |
| `attention_is_all_you_need_neurips_2017.pdf` | [NeurIPS 2017](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need.pdf) | `d87d482d5ae7960e2e43d7dd6d21377e60e73e8fce1bf2a01aff7aca8a08c537` |
| `phan_tich_cau_ghep_tieng_viet_jcc_2012.pdf` | [Tạp chí Tin học và Điều khiển học](https://vjs.ac.vn/jcc/article/download/1451/pdf) | `7de945cdb846db932d6ec76ecb4f405a96030e983eb643fa9d6a016507a0b835` |

Nếu bản tải về khác hash, cần xác nhận lại nội dung và nhãn trang trước khi so sánh kết quả.

```powershell
python scripts/evaluate_system.py --dataset sample_papers/upgrade_pilot.jsonl --orchestration-mode autonomous --max-rounds 2 --output evaluation/upgrade-pilot.json
python -m pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
mypy src/research_assist
npm run test:dom
npm run test:ui:workflow
npm run test:ui:papers
```

Lệnh benchmark trên chạy demo ngoại tuyến trong kho tạm, không gọi LLM. Chỉ thêm `--live` khi muốn dùng provider đã cấu hình. Với browser test, cài Chromium qua Playwright hoặc đặt `UI_CHROMIUM_PATH` đến Chrome; có thể đặt `RESEARCH_PYTHON` đến Python của virtualenv.

## Kết quả đã quan sát

- Kiểm thử cuối ngày 07/11/2026: **461 bài backend qua**, Ruff lint/format qua, mypy qua **71 file nguồn**. DOM/API integration **12** kiểm tra, browser smoke **14**, workflow **19**, và suite tìm/nhập bài báo đều qua. Browser test mới đã được thêm vào GitHub Actions. Đây là kết quả chạy cục bộ; chưa khẳng định CI từ xa đã chạy commit mới.
- Pilot cuối: **20/20** lượt hoàn tất, **0** đoạn truy xuất ngoài phạm vi đã chọn. Recall đoạn tham chiếu trung bình: single **0,8**, multi **0,8**, table **1,0**. Các chỉ số recall này bằng lượt trước nâng cấp; chưa có bằng chứng để tuyên bố tăng độ chính xác tổng thể.
- Demo từ chối **3/5** câu không có đáp án; còn **2/5** trả về đoạn gần từ khóa. Demo chỉ phù hợp thử luồng, chưa đánh giá được ngữ nghĩa đầy đủ. Bộ lọc định danh/đơn vị giúp hạn chế một số trường hợp nhưng không thay cho LLM.
- Smoke test với model thật, chế độ fixed, BM25 không embedding/reranker: câu so sánh Bảng 3 trả lời **42.5%** và **75.1%**, bằng chứng ở trang PDF vật lý **12**, hai nhận định được checker xác nhận, **4** lần gọi LLM, dừng vì đủ yêu cầu. Câu hỏi p-value không có trong tài liệu đã từ chối trả lời ở lượt kiểm tra riêng.
- Một lượt live trước đó đã hết ngân sách sau retry/timeout ở provider với giới hạn 6 lần gọi. Lượt bảng thành công dùng giới hạn 10 lần gọi, 48.000 token, timeout mỗi request 65 giây, timeout run 200 giây; chỉ dùng 4 lần gọi. Các giá trị này chỉ áp dụng cho smoke test, không sửa cấu hình người dùng.
- arXiv thật trả được metadata phiên bản `1706.03762v7` và PDF 2.215.244 byte. UI kiểm tra việc hiển thị metadata như văn bản, thử lại khi nhập lỗi, cập nhật kho, giữ nguyên tập nguồn, kết quả rỗng, giới hạn truy cập và độ rộng 1280/640/390 px. Không có lỗi JavaScript trong lượt kiểm tra.

## Giới hạn và bước đánh giá tiếp theo

Nhận diện bảng hiện tập trung vào bảng có đường kẻ hoặc bảng khoa học có caption và đường kẻ ngang. Bảng scan, bảng hoàn toàn không đường kẻ, ô gộp phức tạp hoặc bảng nhiều trang vẫn có thể cần đối chiếu PDF. Công cụ mở bảng có giới hạn 12 cửa sổ nguồn; không được mặc định rằng một cửa sổ đã chứa đủ tiêu đề và mọi hàng.

Bộ lọc ý nghĩa thống kê là kiểm tra bảo thủ bằng quy tắc trên các trích dẫn của nhận định, không phải mô hình suy luận thống kê hoặc một điểm “scientific rigor”. Từ khóa khớp không chứng minh hai câu có cùng ý nghĩa; Fact Checker vẫn phải xác định đúng đối tượng, phép thử và điều kiện thí nghiệm.

Trước khi công bố accuracy: nhờ người độc lập duyệt nhãn, mở rộng tài liệu và câu không có đáp án, khóa tập test riêng, chạy đối chứng trên cùng model/cấu hình và chấm từng nhận định/trích dẫn. Hai smoke test live và pilot 20 câu chưa đủ để kết luận chất lượng trên mọi tài liệu.
