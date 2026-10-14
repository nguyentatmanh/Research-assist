import argparse
import asyncio
import json

import uvicorn

from research_assist.bootstrap import build_container
from research_assist.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Research Assist — five-agent RAG")
    parser.add_argument(
        "command", choices=["serve", "seed", "reindex", "ask"], nargs="?", default="serve"
    )
    parser.add_argument("question", nargs="?")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    settings = Settings()
    if args.command == "serve":
        uvicorn.run(
            "research_assist.api.app:create_app",
            factory=True,
            host=args.host or settings.host,
            port=args.port or settings.port,
            workers=1,
        )
    else:
        asyncio.run(run_command(args.command, args.question, settings))


async def run_command(command: str, question: str | None, settings: Settings) -> None:
    from research_assist.domain.models import ResearchRequest

    container = build_container(settings)
    try:
        if command == "seed":
            print(
                json.dumps(
                    [d.model_dump() for d in await container.documents.seed()],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif command == "reindex":
            print(f"Updated {await container.documents.reindex()} chunks")
        elif command == "ask":
            if not question:
                raise SystemExit("Usage: research-assist ask 'your research question'")
            run = container.runs.submit(ResearchRequest(question=question))
            await container.runs.tasks[run.id]
            completed = container.repository.run(run.id)
            if not completed.result:
                raise SystemExit(completed.error)
            print(completed.result.markdown)
    finally:
        await container.close()


if __name__ == "__main__":
    main()
