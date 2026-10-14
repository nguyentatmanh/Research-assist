import argparse
import asyncio
from pathlib import Path

from research_assist.bootstrap import build_container
from research_assist.config import Settings


async def ingest(folder: Path) -> None:
    container = build_container(Settings())
    try:
        paths = await asyncio.to_thread(lambda: sorted(folder.iterdir()))
        for path in paths:
            if path.suffix.lower() not in {".txt", ".md", ".pdf", ".docx"} or not path.is_file():
                continue
            data = await asyncio.to_thread(path.read_bytes)
            document, duplicate = await container.documents.ingest(data, path.name)
            print(
                f"{path.name}: {document.chunk_count} chunks "
                f"({'existing' if duplicate else 'created'})"
            )
    finally:
        await container.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a local folder before starting the server")
    parser.add_argument("folder", type=Path)
    asyncio.run(ingest(parser.parse_args().folder))


if __name__ == "__main__":
    main()
