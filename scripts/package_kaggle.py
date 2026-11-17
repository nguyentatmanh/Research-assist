"""Build the importable Kaggle notebook bundle from the notebooks in this checkout."""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOKS = (
    "01_train_embedding.ipynb",
    "02_train_reranker.ipynb",
    "03_gpu_server.ipynb",
    "04_kaggle_workspace.ipynb",
    "06_kaggle_llm_worker.ipynb",
)


def build() -> Path:
    notebooks = ROOT / "notebooks"
    files = [(name, (notebooks / name).read_bytes()) for name in BOOKS]
    master = ROOT / "kaggle" / "kaggle_master_worker.ipynb"
    files.append((master.name, master.read_bytes()))
    files.append(("boot.sh", (notebooks / "boot.sh").read_bytes()))
    guide = (ROOT / "docs/KAGGLE.md").read_text(encoding="utf-8")
    guide = guide.replace("](../notebooks/", "](")
    guide = guide.replace(f"](../kaggle/{master.name})", f"]({master.name})")
    files.append(("KAGGLE.md", guide.encode("utf-8")))
    output = notebooks / "kaggle_bundle.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files:
            # Keep identical inputs byte-for-byte reproducible.
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED)
    return output


if __name__ == "__main__":
    print(build())
