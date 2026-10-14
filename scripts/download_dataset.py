"""Download official QASPER through its parquet conversion; keep all original splits."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--output-dir", type=Path, default=Path("data/datasets/qasper"))
    parser.add_argument("--revision", default="")
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    # The original repo contains a legacy loading script, unsupported by datasets >= 4.
    # Use the official Hub parquet conversion, pinned to its commit, without remote code.
    revision = (
        args.revision or HfApi().dataset_info("allenai/qasper", revision="refs/convert/parquet").sha
    )
    files = HfApi().list_repo_files("allenai/qasper", repo_type="dataset", revision=revision)
    data_files = {
        split: [
            hf_hub_download("allenai/qasper", file, repo_type="dataset", revision=revision)
            for file in files
            if file.endswith(".parquet") and f"/{split}/" in file
        ]
        for split in ("train", "validation", "test")
    }
    if any(not paths for paths in data_files.values()):
        raise ValueError("Missing parquet split; check the official dataset conversion")
    dataset = load_dataset("parquet", data_files=data_files)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        path = args.output / f"{split}.json"
        path.write_text(
            json.dumps({row["id"]: row for row in dataset[split]}, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"{split}: {len(dataset[split])} papers -> {path}")
    (args.output / "download.json").write_text(
        json.dumps(
            {
                "dataset": "allenai/qasper",
                "revision": revision,
                "license": "CC-BY-4.0",
                "url": "https://huggingface.co/datasets/allenai/qasper",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
