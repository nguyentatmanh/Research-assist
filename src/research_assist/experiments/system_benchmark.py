"""Run the actual five-agent pipeline in an isolated workspace; never invent answer accuracy."""

import argparse
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from research_assist.bootstrap import build_container
from research_assist.config import Settings
from research_assist.domain.models import Contract, ResearchRequest


class Case(Contract):
    id: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=8, max_length=2000)
    requirements: list[str] = Field(default_factory=list, max_length=6)
    files: list[str] = Field(min_length=1, max_length=50)
    relevant_sources: list[str] = Field(default_factory=list)
    answerable: bool | None = None
    label_source: Literal["synthetic", "human", "unreviewed"] = "unreviewed"
    language: Literal["en", "vi"] = "en"


def load_cases(path: Path) -> list[Case]:
    cases = [
        Case.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and have unique IDs")
    for case in cases:
        for name in case.files:
            file = (path.parent / name).resolve()
            if not file.is_relative_to(path.parent.resolve()) or not file.is_file():
                raise ValueError("Dataset files must exist below the case-file directory")
        if not set(case.relevant_sources).issubset({Path(f).name for f in case.files}):
            raise ValueError("Relevant sources must name files inside the case scope")
    return cases


async def benchmark(
    dataset: Path, *, live: bool, profile: str, max_rounds: int
) -> dict[str, object]:
    cases = await asyncio.to_thread(load_cases, dataset)
    dataset_bytes = await asyncio.to_thread(dataset.read_bytes)
    document_hashes: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="research-benchmark-") as directory:
        options: dict[str, object] = {
            "data_dir": Path(directory),
            "cache_enabled": False,
            "workflow_profile": profile,
        }
        if not live:
            options.update(
                llm_provider="demo",
                llm_model="",
                agent_routes={},
                embedding_provider="none",
                reranker_model="",
                chunk_strategy="chars",
            )
        settings = Settings(**options)  # type: ignore[arg-type]
        container = build_container(settings)
        try:
            documents = {}
            for name in dict.fromkeys(name for case in cases for name in case.files):
                content = await asyncio.to_thread((dataset.parent / name).read_bytes)
                document_hashes[name] = hashlib.sha256(content).hexdigest()
                document, _ = await container.documents.ingest(content, Path(name).name)
                documents[name] = document.id
            rows: list[dict[str, object]] = []
            for case in cases:
                request = ResearchRequest(
                    question=case.question,
                    language=case.language,
                    requirements=case.requirements,
                    document_ids=list(dict.fromkeys(documents[f] for f in case.files)),
                    max_rounds=max_rounds,
                    use_cache=False,
                )
                run = container.runs.submit(request)
                await container.runs.tasks[run.id]
                finished = container.repository.run(run.id)
                row: dict[str, object] = {
                    "id": case.id,
                    "question": case.question,
                    "status": finished.status,
                    "label_source": case.label_source,
                    "answerable": case.answerable,
                }
                if finished.result:
                    result = finished.result
                    found = {chunk.filename for chunk in result.retrieved}
                    row.update(
                        result=result.model_dump(),
                        answer_sha256=hashlib.sha256(result.markdown.encode()).hexdigest(),
                        source_recall=(
                            len(found.intersection(case.relevant_sources))
                            / len(case.relevant_sources)
                            if case.relevant_sources
                            else None
                        ),
                        abstained=not result.reported_claim_ids,
                        human_faithfulness=None,
                        human_answer_completeness=None,
                    )
                else:
                    row["error"] = finished.error
                rows.append(row)
            completed = [row for row in rows if row["status"] == "completed"]
            elapsed = [row["result"]["elapsed_ms"] for row in completed]  # type: ignore[index]
            return {
                "format_version": 1,
                "kind": "system_benchmark",
                "policy_version": "2.1",
                "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                "document_sha256": document_hashes,
                "configuration": {
                    "live": live,
                    "profile": profile,
                    "max_rounds": max_rounds,
                    "provider": container.router.name,
                    "model": container.router.model,
                    "embedding_fingerprint": container.embedder.fingerprint,
                    "reranker": settings.reranker_model or None,
                },
                "summary": {
                    "cases": len(rows),
                    "completed": len(completed),
                    "latency_p50_ms": float(np.percentile(elapsed, 50)) if elapsed else None,
                    "latency_p95_ms": float(np.percentile(elapsed, 95)) if elapsed else None,
                    "human_faithfulness": None,
                    "human_answer_completeness": None,
                },
                "notes": [
                    "Source recall is file-level, not passage recall or answer accuracy.",
                    "No independent human answer labels were scored.",
                    "Demo validates plumbing only. Cache is disabled; "
                    "warm local model/index reuse is allowed.",
                ],
                "cases": rows,
            }
        finally:
            await container.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("examples/benchmark/cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/system.json"))
    parser.add_argument(
        "--live", action="store_true", help="Use configured providers/models; may incur API charges"
    )
    parser.add_argument("--profile", choices=["balanced", "academic"], default="balanced")
    parser.add_argument("--max-rounds", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    result = asyncio.run(
        benchmark(
            args.dataset.resolve(), live=args.live, profile=args.profile, max_rounds=args.max_rounds
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}; model quality still requires independent human review.")


if __name__ == "__main__":
    main()
