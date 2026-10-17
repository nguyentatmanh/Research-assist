"""Build a source handover with an integrity manifest and no local runtime data."""

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

ROOT_FILES = {
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "ARCHITECTURE.md",
    "DELIVERY_NOTES.md",
    "Dockerfile",
    "IMPLEMENTATION_NOTES.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "SOURCES.md",
    "compose.yaml",
    "package-lock.json",
    "package.json",
    "pyproject.toml",
    "requirements-tested.txt",
}
SOURCE_DIRS = {".github", "configs", "docs", "examples", "notebooks", "scripts", "src", "tests"}
EXCLUDED_DIRS = {"__pycache__", "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def source_files(root: Path) -> list[Path]:
    files = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in sorted(SOURCE_DIRS):
        for folder, dirs, names in os.walk(root / name):
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in EXCLUDED_DIRS
                and not d.endswith(".egg-info")
                and not (Path(folder) / d).is_symlink()
            )
            for filename in sorted(names):
                path = Path(folder) / filename
                if path.is_symlink() or path.suffix in {".pyc", ".pyo", ".env"}:
                    continue
                if ".env." in filename and not filename.endswith(".env.example"):
                    continue
                if filename.startswith(".env") and filename != ".env.example":
                    continue
                files.append(path)
    if any(not path.resolve().is_relative_to(root) for path in files):
        raise ValueError("Source file escapes project root")
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def package_source(root: Path, output: Path) -> int:
    root = root.resolve()
    files = source_files(root)
    manifest = root / "SOURCE_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "project": "Research Desk",
                "algorithm": "sha256",
                "files": {
                    path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in files
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in [*files, manifest]:
            archive.write(path, "research-assist/" + path.relative_to(root).as_posix())
    return len(files) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/ResearchDesk_Source.zip"))
    args = parser.parse_args()
    count = package_source(Path(__file__).resolve().parents[1], args.output)
    print(f"Packaged {count} files: {args.output}")


if __name__ == "__main__":
    main()
