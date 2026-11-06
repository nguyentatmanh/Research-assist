"""Notebook bootstrap: verify a fresh process and stop before using stale native modules."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

TRAINING_REQUIREMENTS = [
    "sentence-transformers==5.1.2",
    "transformers==4.57.1",
    "datasets==4.3.0",
    "accelerate==1.11.0",
    "huggingface-hub==0.36.0",
    "pyarrow==21.0.0",
    "numpy==2.2.6",
    "sentencepiece==0.2.1",
]
WATCHED_MODULES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "torch": "torch",
    "transformers": "transformers",
    "sentence_transformers": "sentence-transformers",
    "datasets": "datasets",
    "huggingface_hub": "huggingface-hub",
    "tokenizers": "tokenizers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
}
PREFLIGHT_CODE = r"""
import json, sys
import numpy as np
from numpy import strings
assert strings.add(["a"], ["b"]).tolist() == ["ab"]
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
import pandas, pyarrow, datasets, torch
from sentence_transformers import (
    SentenceTransformer, CrossEncoder,
    SentenceTransformerTrainer, SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
assert csr_matrix([[1.0]]).shape == (1, 1)
assert pairwise_distances([[0.0], [1.0]]).shape == (2, 2)
print(json.dumps({"imports_ok": True, "python": sys.version.split()[0],
                  "numpy": np.__version__, "numpy_file": np.__file__,
                  "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}))
"""


def installed_versions():
    versions = {}
    for name in set(WATCHED_MODULES.values()) | {r.split("==")[0] for r in TRAINING_REQUIREMENTS}:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def loaded_versions():
    return {
        name: vars(sys.modules[name]).get("__version__")
        for name in WATCHED_MODULES
        if name in sys.modules and sys.modules[name] is not None
    }


def current_runtime_identity():
    # A restarted container/kernel may reuse the same PID. A token stored on sys
    # survives re-running the cell, but cannot survive a real Python restart.
    if not getattr(sys, "_researchdesk_runtime_id", None):
        sys._researchdesk_runtime_id = f"{os.getpid()}:{uuid.uuid4().hex}"
    return sys._researchdesk_runtime_id


def install_training_environment(marker_path, runtime_id=None):
    """Never treat a successful pip install as proof the current kernel is coherent."""
    runtime_id = runtime_id or current_runtime_identity()
    marker_path = Path(marker_path)
    before, loaded = installed_versions(), loaded_versions()
    needed = [r for r in TRAINING_REQUIREMENTS if before[r.split("==")[0]] != r.split("==")[1]]
    if needed:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--prefer-binary",
                "--only-binary=numpy,pyarrow",
                "--upgrade-strategy",
                "only-if-needed",
                *TRAINING_REQUIREMENTS,
            ],
            check=True,
        )
    # A child process has no old NumPy C-extension objects from the notebook kernel.
    checked = subprocess.run(
        [sys.executable, "-c", PREFLIGHT_CODE],
        capture_output=True,
        text=True,
        timeout=180,
    )
    repaired = False
    diagnostics = (checked.stdout or "") + (checked.stderr or "")
    if checked.returncode and any(
        text in diagnostics
        for text in (
            "numpy.ufunc",
            "numpy.dtype size changed",
            "numpy._core",
            "numpy.core",
        )
    ):
        # Repair only NumPy's files; do not reinstall Kaggle's CUDA PyTorch.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--force-reinstall",
                "--no-cache-dir",
                "--no-deps",
                "--only-binary=:all:",
                "numpy==2.2.6",
            ],
            check=True,
        )
        repaired = True
        checked = subprocess.run(
            [sys.executable, "-c", PREFLIGHT_CODE],
            capture_output=True,
            text=True,
            timeout=180,
        )
    if checked.returncode:
        print((checked.stdout or "")[-4000:])
        print((checked.stderr or "")[-12000:])
        raise RuntimeError(
            "Fresh-process dependency check failed. Training has NOT started. "
            "Keep the complete output above for diagnosis."
        )
    after = installed_versions()
    fingerprint = hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest()
    previous = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.exists() else {}
    changed = before != after
    mismatch = any(
        version is not None and str(version) != str(after.get(WATCHED_MODULES[name]))
        for name, version in loaded.items()
    )
    pending_same_kernel = previous.get("restart_required_for") == runtime_id
    unknown_loaded_kernel = bool(loaded) and (
        previous.get("fingerprint") != fingerprint
        or previous.get("runtime_identity_format") != "memory-token-v1"
    )
    restart = (
        pending_same_kernel
        or bool(loaded)
        and (changed or repaired or mismatch or unknown_loaded_kernel)
    )
    status = {
        "ready": not restart,
        "fingerprint": fingerprint,
        "versions": after,
        "runtime_id": runtime_id,
        "runtime_identity_format": "memory-token-v1",
        "restart_required_for": runtime_id if restart else None,
        "loaded_modules_before_install": sorted(loaded),
        "fresh_process_output": checked.stdout.strip(),
        "numpy_repaired": repaired,
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(checked.stdout.strip())
    if restart:
        print(
            "THIẾT LẬP ĐÃ THÀNH CÔNG — cần RESTART KERNEL/PYTHON một lần.\n"
            "Đây là bước hoàn tất cài đặt, không phải lỗi huấn luyện.\n"
            "Sau restart, chạy lại từ cell đầu. Các cell phía sau sẽ tạm bỏ qua "
            "để tránh dùng thư viện cũ trong RAM."
        )
        return status
    print("Environment ready: fresh-process imports passed and no kernel restart is pending.")
    return status
