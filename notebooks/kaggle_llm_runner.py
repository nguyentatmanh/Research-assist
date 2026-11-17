"""Install and run the isolated vLLM worker from the standalone Kaggle notebook."""

import atexit
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
from pathlib import Path

VLLM_VERSION = "0.10.2"
REQUIREMENTS = [
    f"vllm=={VLLM_VERSION}",
    "transformers==4.56.1",
    "openai==1.107.2",
    "httpx==0.28.1",
]
PROFILES = {
    "7b": ("Qwen/Qwen2.5-7B-Instruct-AWQ", "b25037543e9394b818fdfca67ab2a00ecc7dd641", 1, 16384),
    "32b": ("Qwen/Qwen2.5-32B-Instruct-AWQ", "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c", 2, 8192),
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def request_json(url, path, *, key=None, body=None, timeout=5):
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url + path, data=data, headers=headers)
    with HTTP.open(request, timeout=timeout) as response:
        return json.load(response)


def check_authentication(url):
    try:
        request_json(url, "/v1/models")
    except urllib.error.HTTPError as error:
        if error.code == 401:
            return
        raise
    raise RuntimeError("Worker unexpectedly responded without a token.")


def wait_for_catalog(url, model, key, processes, log, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError(f"A worker process stopped. See {log} for details.")
        try:
            data = request_json(url, "/v1/models", key=key)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 502, 503, 504, 530):
                raise RuntimeError(
                    f"Model catalog returned HTTP {error.code}. See {log}."
                ) from None
        except (urllib.error.URLError, TimeoutError):
            pass
        else:
            if any(item["id"] == model for item in data["data"]):
                return
            raise RuntimeError(f"The worker catalog did not contain {model}.")
        time.sleep(2)
    raise TimeoutError(f"The model server did not become ready. See {log}.")


def check_structured_output(url, model, key):
    expected = {"ready": "worker_ready"}
    sample = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return a JSON object."},
            {"role": "user", "content": 'Return {"other": "unexpected"}.'},
        ],
        "temperature": 0,
        "max_tokens": 64,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "worker_probe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"ready": {"type": "string", "enum": ["worker_ready"]}},
                    "required": ["ready"],
                    "additionalProperties": False,
                },
            },
        },
    }
    response = request_json(url, "/v1/chat/completions", key=key, body=sample, timeout=90)
    if json.loads(response["choices"][0]["message"]["content"]) != expected:
        raise RuntimeError("The worker did not follow the requested JSON schema.")


def available_port():
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def stop_worker_tree(process):
    # vLLM tensor parallelism launches children in the worker process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def stop_llm_worker(stop_tunnel):
    tunnel = getattr(sys, "_research_assist_llm_tunnel", None)
    if tunnel is not None:
        stop_tunnel(tunnel)
    sys._research_assist_llm_tunnel = None
    for process in reversed(getattr(sys, "_research_assist_llm_processes", [])):
        stop_worker_tree(process)
    sys._research_assist_llm_processes = []


def worker_token():
    try:
        from kaggle_secrets import UserSecretsClient

        key = UserSecretsClient().get_secret("RESEARCH_GPU_API_KEY").strip()
    except Exception:
        raise RuntimeError(
            "Create RESEARCH_GPU_API_KEY in Kaggle Secrets and enable notebook access."
        ) from None
    if not re.fullmatch(r"[!-~]{32,256}", key):
        raise ValueError(
            "RESEARCH_GPU_API_KEY must contain 32–256 printable ASCII characters without spaces."
        )
    return key


def prepare_vllm(directory):
    environment = directory / f"venv-vllm-{VLLM_VERSION}"
    python = environment / "bin" / "python"
    if not python.is_file():
        venv.create(environment, with_pip=True)
    marker = environment / "research-assist-requirements.json"
    requirements = json.dumps(REQUIREMENTS)
    if not marker.is_file() or marker.read_text(encoding="utf-8") != requirements:
        print(f"Cài đặt vLLM {VLLM_VERSION} trong môi trường riêng…", flush=True)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--only-binary=:all:",
                *REQUIREMENTS,
            ],
            check=True,
            timeout=1800,
        )
        marker.write_text(requirements, encoding="utf-8")
    return python


def run_llm_worker(profile, public_tunnel, gateway_source, *, start_tunnel, stop_tunnel):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Use a Kaggle Linux x86-64 notebook.")
    if sys.version_info[:2] not in {(3, 10), (3, 11), (3, 12)}:
        raise RuntimeError("This vLLM worker requires Python 3.10–3.12.")
    if profile not in PROFILES:
        raise ValueError('Set MODEL_PROFILE to "7b" or "32b".')
    key = worker_token()
    model, revision, tensor_parallel, context_length = PROFILES[profile]
    directory = Path("/kaggle/temp/research-assist-llm")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        gpu_info = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError("Enable an NVIDIA GPU accelerator in Kaggle.") from None
    if gpu_info.returncode or len(gpu_info.stdout.splitlines()) < tensor_parallel:
        raise RuntimeError(f"Enable an Accelerator with at least {tensor_parallel} visible GPU(s).")
    print(f"GPU: {', '.join(gpu_info.stdout.splitlines())}")
    stop_llm_worker(stop_tunnel)
    if not getattr(sys, "_research_assist_llm_exit_registered", False):
        atexit.register(stop_llm_worker, stop_tunnel)
        sys._research_assist_llm_exit_registered = True

    python = prepare_vllm(directory)
    env = os.environ.copy()
    env.pop("RESEARCH_GPU_API_KEY", None)
    env.update(
        VLLM_USE_V1="0",
        VLLM_ATTENTION_BACKEND="XFORMERS",
        VLLM_NO_USAGE_STATS="1",
        HF_HOME=str(directory / "hf"),
        PYTHONNOUSERSITE="1",
    )
    cuda_check = (
        "import torch, sys; n = int(sys.argv[1]); "
        "assert torch.cuda.device_count() >= n, 'Not enough CUDA devices'; "
        "assert all(torch.cuda.get_device_capability(i) >= (7,5) for i in range(n)), "
        "'AWQ requires SM 7.5 or newer'; "
        "assert all(float(torch.ones(1, device=f'cuda:{i}').item()) == 1 for i in range(n))"
    )
    subprocess.run(
        [str(python), "-c", cuda_check, str(tensor_parallel)], env=env, check=True, timeout=120
    )
    upstream = available_port()
    port = available_port()
    while port == upstream:
        port = available_port()
    local_url = f"http://127.0.0.1:{port}"
    gateway_path = directory / "gateway.py"
    gateway_path.write_text(gateway_source, encoding="utf-8")
    log_path = directory / "worker.log"
    try:
        print(f"Đang tải và khởi động {model} (TP={tensor_parallel})…", flush=True)
        command = [
            str(python),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--revision",
            revision,
            "--load-format",
            "safetensors",
            "--host",
            "127.0.0.1",
            "--port",
            str(upstream),
            "--tensor-parallel-size",
            str(tensor_parallel),
            "--distributed-executor-backend",
            "mp",
            "--dtype",
            "float16",
            "--quantization",
            "awq",
            "--enforce-eager",
            "--max-model-len",
            str(context_length),
            "--max-num-seqs",
            "2",
            "--gpu-memory-utilization",
            "0.90" if tensor_parallel == 1 else "0.94",
            "--guided-decoding-backend",
            "xgrammar",
            "--generation-config",
            "vllm",
            "--disable-log-requests",
        ]
        if tensor_parallel > 1:
            command += ["--disable-custom-all-reduce"]
        processes = sys._research_assist_llm_processes
        with log_path.open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            processes.append(server)
            gateway = subprocess.Popen(
                [str(python), str(gateway_path), str(upstream), str(port)],
                env={**env, "RESEARCH_GPU_API_KEY": key},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(gateway)
        print(f"Xem tiến trình tải model tại {log_path}")
        wait_for_catalog(local_url, model, key, processes, log_path, timeout=1800)
        check_authentication(local_url)
        check_structured_output(local_url, model, key)
        print("Worker đã xác nhận API Bearer và phản hồi JSON theo schema.")

        url = local_url
        if public_tunnel:
            tunnel, url = start_tunnel(port, directory / "tunnel")
            sys._research_assist_llm_tunnel = tunnel
            wait_for_catalog(url, model, key, [*processes, tunnel], log_path, timeout=70)
            check_authentication(url)
        print(f"Provider = OpenAI-compatible\nEndpoint /v1 = {url}/v1\nModel = {model}")
        print("API key: dùng giá trị RESEARCH_GPU_API_KEY đã lưu trong Kaggle Secrets.")
    except BaseException:
        stop_llm_worker(stop_tunnel)
        raise
