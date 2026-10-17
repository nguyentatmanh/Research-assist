"""Pin a public Hugging Face checkpoint and save a self-contained local model directory."""

import argparse
import json
from pathlib import Path


def main() -> None:
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--output", type=Path, default=Path("models/multilingual-e5-small"))
    parser.add_argument("--revision", default="")
    parser.add_argument("--family", choices=["e5", "generic", "qwen3"], default="e5")
    args = parser.parse_args()
    revision = args.revision or HfApi().model_info(args.model).sha
    files = HfApi().list_repo_files(args.model, revision=revision)
    exclude_bin = ["*.bin"] if any(f.endswith(".safetensors") for f in files) else []
    snapshot_download(
        repo_id=args.model,
        revision=revision,
        local_dir=args.output,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "README.md"],
        ignore_patterns=["onnx/*", "openvino/*", "*.h5", "*.ot", "*.msgpack", *exclude_bin],
    )
    (args.output / "research_model.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "model_family": args.family,
                "base_model": args.model,
                "revision": revision,
                "fine_tuned": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {args.model} @ {revision} to {args.output}")


if __name__ == "__main__":
    main()
