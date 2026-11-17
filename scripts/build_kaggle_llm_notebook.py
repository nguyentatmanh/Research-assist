"""Generate the standalone Kaggle vLLM worker notebook from reviewed Python sources."""

import ast
import json
import textwrap
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def notebook_cell(kind, cell_id, source):
    source = textwrap.dedent(source).strip()
    cell = {
        "cell_type": kind,
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        ast.parse(source, filename=cell_id)
        cell.update(execution_count=None, outputs=[])
    return cell


def build_notebook():
    introduction = """
    # Research Assist · Worker LLM trên Kaggle

    Notebook phục vụ model sinh câu trả lời cho Research Assist đang chạy trên thiết bị của
    bạn. Chọn **Accelerator → GPU** và **Internet → On** trong Kaggle.
    Chọn `7b` để chạy trên một T4 hoặc GPU tương thích. Chế độ thử nghiệm `32b`
    cần **GPU T4 x 2**, nạp hai phân đoạn trọng số bằng tensor parallelism.

    1. Mở **Add-ons → Secrets**, tạo `RESEARCH_GPU_API_KEY` bằng một chuỗi ngẫu nhiên
       dài ít nhất 32 ký tự, rồi cho notebook truy cập secret.
    2. Đặt `START_LLM_WORKER = True` trong ô cấu hình. Bật thêm
       `OPEN_PUBLIC_TUNNEL` nếu Research Assist chạy ngoài Kaggle. Chạy các ô đến **Khởi động
       worker**. Lần đầu cần tải thư viện và trọng số.
    3. Khi notebook xác nhận phản hồi JSON đúng schema, sao chép **Endpoint /v1** và **Model**.
       Trong Research Assist, mở cấu hình model → **GPU Worker** → provider
       **OpenAI-compatible**. Nhập Endpoint, Model, cùng khóa đã tạo ở bước 1.
       Lưu, kiểm tra kết nối, sau đó chọn GPU cho agent muốn sử dụng.

    Giữ phiên Kaggle chạy khi đang dùng worker. Link Cloudflare là tạm thời và thay đổi
    mỗi lần mở lại. Khi cần đổi cấu hình, sửa `MODEL_PROFILE` và chạy lại đến ô khởi động.
    Mở ô dừng riêng ở cuối khi đã dùng xong. Run All tự giữ worker chạy cho đến khi
    bạn dừng ô khởi động hoặc kết thúc phiên.

    - vLLM 0.10.2 dùng môi trường Python riêng và wheel CUDA 12.8, cần driver tương thích.
      Cấu hình AWQ dùng FP16 và XFormers cho Turing. Thông tin lỗi GPU và nạp model nằm
      trong log được in ở lúc khởi động. Nếu profile 32B không đủ VRAM trên phiên
      đang dùng, chọn `7b`.
    - Giới hạn ngữ cảnh gồm cả đầu vào và phần sinh ra. Nếu model báo vượt giới hạn,
      giảm độ dài đầu vào hoặc số token đầu ra của agent.
    - Đầu vào suy luận đi từ Research Assist qua tunnel tới GPU. Chỉ dùng dữ liệu
      bạn có thể gửi qua kết nối đó. Token đọc từ Kaggle Secrets và không in ra output.

    Nguồn kỹ thuật [vLLM GPU](https://docs.vllm.ai/en/v0.10.2/getting_started/installation/gpu.html),
    [vLLM Structured Outputs](https://docs.vllm.ai/en/v0.10.2/features/structured_outputs.html),
    [Qwen 7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ) và
    [Qwen 32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ).
    """
    config = """
    START_LLM_WORKER = False
    MODEL_PROFILE = "7b"  # "32b" requires two GPUs
    OPEN_PUBLIC_TUNNEL = False  # Set True for Research Assist running outside Kaggle.
    KEEP_WORKER_RUNNING = True   # Set False to return the control cell immediately.
    """
    helpers = (NOTEBOOKS / "kaggle_tunnel.py").read_text(encoding="utf-8")
    runner = (NOTEBOOKS / "kaggle_llm_runner.py").read_text(encoding="utf-8")
    gateway = (NOTEBOOKS / "kaggle_llm_gateway.py").read_text(encoding="utf-8")
    start = """
    if not START_LLM_WORKER:
        print("Đặt START_LLM_WORKER = True và chạy lại từ ô cấu hình để mở worker.")
    else:
        run_llm_worker(
            MODEL_PROFILE, OPEN_PUBLIC_TUNNEL, GATEWAY_SOURCE,
            start_tunnel=start_quick_tunnel, stop_tunnel=stop_process,
        )
        if KEEP_WORKER_RUNNING:
            print("Worker đang chạy. Ngắt ô này để dừng.", flush=True)
            try:
                children = list(sys._research_assist_llm_processes)
                if sys._research_assist_llm_tunnel is not None:
                    children.append(sys._research_assist_llm_tunnel)
                while all(process.poll() is None for process in children):
                    time.sleep(2)
                print("Một tiến trình worker đã dừng. Xem log ở đường dẫn in phía trên.")
            except KeyboardInterrupt:
                print("Đang dừng worker…")
            finally:
                stop_llm_worker(stop_process)
    """
    cells = [
        notebook_cell("markdown", "intro", introduction),
        notebook_cell("code", "config", config),
        notebook_cell("code", "tunnel-helper", helpers),
        notebook_cell("code", "worker-runner", runner),
        notebook_cell("code", "gateway-source", "GATEWAY_SOURCE = " + repr(gateway)),
        notebook_cell("markdown", "start-label", "## Khởi động worker"),
        notebook_cell("code", "start", start),
        notebook_cell(
            "markdown",
            "stop-label",
            "## Dừng worker\nBật `STOP_LLM_WORKER` và chạy ô dưới khi cần dừng worker.",
        ),
        notebook_cell(
            "code",
            "stop",
            "STOP_LLM_WORKER = False\nif STOP_LLM_WORKER:\n    stop_llm_worker(stop_process)",
        ),
    ]
    for cell in cells:
        if cell["id"] in {"tunnel-helper", "worker-runner", "gateway-source"}:
            cell["metadata"] = {"jupyter": {"source_hidden": True}}
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    output = NOTEBOOKS / "06_kaggle_llm_worker.ipynb"
    output.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
