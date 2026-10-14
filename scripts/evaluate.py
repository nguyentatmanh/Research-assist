"""Reproducible document-level retrieval evaluation and optional full pipeline audit."""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from research_assist.bootstrap import build_container
from research_assist.config import Settings
from research_assist.domain.models import ResearchRequest
from research_assist.infrastructure.repository import now
from research_assist.infrastructure.search import HybridSearch


async def evaluate(args: argparse.Namespace) -> None:
    settings = Settings()
    container = build_container(settings)
    cases = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    try:
        if args.seed:
            await container.documents.seed()
        if not container.repository.documents():
            raise SystemExit(
                "Ingest your benchmark documents first or use --seed for the smoke set"
            )
        search = HybridSearch(container.repository, container.embedder, settings)
        modes = ["lexical", "hybrid"] if args.mode == "both" else [args.mode]
        rows = []
        for mode in modes:
            for case in cases:
                started = time.perf_counter()
                ids = case.get("document_ids", [])
                if case.get("scope_files"):
                    mapping = {d.filename: d.id for d in container.repository.documents()}
                    ids = [mapping[name] for name in case["scope_files"]]
                retrieved, warnings = await search.search([case["question"]], ids, args.top_k, mode)
                expected = set(case["relevant_files"])
                found = {c.filename for c in retrieved}
                ranks = [rank for rank, c in enumerate(retrieved, 1) if c.filename in expected]
                row = {
                    "id": case["id"],
                    "mode": mode,
                    "question": case["question"],
                    "unanswerable": bool(case.get("unanswerable", False)),
                    "doc_recall_at_chunk_k": len(expected & found) / len(expected)
                    if expected
                    else None,
                    "doc_mrr_at_chunk_k": 1 / min(ranks) if ranks else 0 if expected else None,
                    "retrieved_files": [c.filename for c in retrieved],
                    "retrieval_elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "warnings": warnings,
                }
                if args.pipeline:
                    run = container.runs.submit(
                        ResearchRequest(
                            question=case["question"],
                            retrieval_mode=mode,
                            document_ids=ids,
                            top_k=args.top_k,
                            max_rounds=args.max_rounds,
                        )
                    )
                    await container.runs.tasks[run.id]
                    finished = container.repository.run(run.id)
                    row["pipeline_status"] = finished.status
                    if finished.result:
                        result = finished.result
                        supported = sum(check.verdict == "supported" for check in result.checks)
                        row.update(
                            {
                                "claim_support_rate": supported / len(result.claims)
                                if result.claims
                                else None,
                                "abstained": supported == 0,
                                "llm_calls": result.llm_calls,
                                "input_tokens": result.input_tokens,
                                "output_tokens": result.output_tokens,
                                "pipeline_elapsed_ms": result.elapsed_ms,
                                "run_id": run.id,
                            }
                        )
                    else:
                        row["error"] = finished.error
                rows.append(row)
        summaries = []
        for mode in modes:
            group = [row for row in rows if row["mode"] == mode]
            answerable = [row for row in group if row["doc_recall_at_chunk_k"] is not None]
            unanswerable = [row for row in group if row["unanswerable"] and "abstained" in row]
            summaries.append(
                {
                    "mode": mode,
                    "cases": len(group),
                    "mean_doc_recall_at_chunk_k": statistics.mean(
                        row["doc_recall_at_chunk_k"] for row in answerable
                    )
                    if answerable
                    else None,
                    "mean_doc_mrr_at_chunk_k": statistics.mean(
                        row["doc_mrr_at_chunk_k"] for row in answerable
                    )
                    if answerable
                    else None,
                    "unanswerable_abstention_rate": statistics.mean(
                        row["abstained"] for row in unanswerable
                    )
                    if unanswerable
                    else None,
                    "effective_semantic_enabled": mode == "hybrid"
                    and container.embedder.fingerprint is not None,
                }
            )
        output = {
            "created_at": now(),
            "dataset": str(args.dataset),
            "top_k_chunks": args.top_k,
            "provider": settings.llm_provider,
            "model": settings.resolved_model,
            "embedding_model": container.embedder.fingerprint,
            "notice": "Smoke set is for plumbing verification, not research accuracy. "
            "Claim support rate is the pipeline judge's verdict, not gold factual accuracy.",
            "summary": summaries,
            "cases": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        print(f"Saved {args.output}")
    finally:
        await container.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/smoke_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation-results/smoke.json"))
    parser.add_argument("--seed", action="store_true")
    parser.add_argument(
        "--pipeline", action="store_true", help="May spend API tokens in a real provider mode"
    )
    parser.add_argument("--mode", choices=["lexical", "hybrid", "both"], default="both")
    parser.add_argument("--top-k", type=int, choices=range(2, 13), default=8)
    parser.add_argument("--max-rounds", type=int, choices=[1, 2], default=2)
    asyncio.run(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()
