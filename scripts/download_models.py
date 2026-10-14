"""Download a pinned multilingual E5 checkpoint for local inference."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-base",
        help="Hugging Face model ID (default: intfloat/multilingual-e5-base)",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the model (default: models/<model-name>)",
    )
    parser.add_argument(
        "--revision",
        default="",
        help="Commit SHA or revision of the model",
    )
    parser.add_argument(
        "--family",
        choices=["e5", "generic", "qwen3"],
        default="e5",
        help="Model family (e5, generic, qwen3)",
    )
    args = parser.parse_args()

    # Tự động tạo thư mục riêng cho từng model nếu không truyền --output
    if args.output is None:
        model_name = args.model.strip("/").split("/")[-1]
        output_dir = Path("models") / model_name
    elif args.output == Path("models") or str(args.output).rstrip("/\\") == "models":
        model_name = args.model.strip("/").split("/")[-1]
        output_dir = args.output / model_name
    else:
        output_dir = args.output

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()

    # Resolve a branch/tag to a fixed commit for reproducibility.
    model_info = api.model_info(
        args.model,
        revision=args.revision or None,
    )
    revision = model_info.sha
    if not revision:
        raise RuntimeError("Không xác định được commit của model.")

    files = api.list_repo_files(args.model, revision=revision)
    has_safetensors = any(
        filename.endswith(".safetensors") for filename in files
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Commit: {revision}")
    print(f"Thư mục lưu model: {output_dir.resolve()}")
    print("Đang tải model...")

    snapshot_download(
        repo_id=args.model,
        revision=revision,
        local_dir=output_dir,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.bin",
            "*.model",
            "*.txt",
            "README.md",
        ],
        ignore_patterns=[
            "onnx/*",
            "openvino/*",
            "*.h5",
            "*.ot",
            "*.msgpack",
            *(["*.bin"] if has_safetensors else []),
        ],
    )

    # Metadata used by Research Assist to identify the model family.
    manifest = {
        "format_version": 2,
        "model_family": args.family,
        "base_model": args.model,
        "revision": revision,
        "fine_tuned": False,
    }
    (output_dir / "research_model.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Tải thành công: {output_dir.resolve()}")


if __name__ == "__main__":
    main()