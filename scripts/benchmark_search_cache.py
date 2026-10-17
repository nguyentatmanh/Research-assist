"""Synthetic CPU timing only. Does not measure model quality or GPU performance."""

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from research_assist.config import Settings
from research_assist.domain.models import Document, IndexedChunk
from research_assist.infrastructure.embeddings import NoEmbeddings
from research_assist.infrastructure.repository import Repository
from research_assist.infrastructure.search import HybridSearch


async def measure(count: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as directory:
        settings = Settings(
            _env_file=None,
            data_dir=Path(directory),
            llm_provider="demo",
            embedding_provider="none",
            agent_routes={},
        )
        repo = Repository(settings.data_dir)
        doc = Document(
            id="doc_bench",
            filename="synthetic.txt",
            title="Synthetic timing fixture",
            kind="txt",
            sha256="synthetic",
            page_count=0,
            chunk_count=count,
            created_at="fixture",
        )
        chunks = []
        for i in range(count):
            text = (
                f"Synthetic topic{i % 97} document{i} studies retrieval evidence "
                f"method{i % 23} with measured latency and citations."
            )
            chunks.append(
                IndexedChunk(
                    id=f"ch_{i:08}",
                    document_id=doc.id,
                    document_title=doc.title,
                    filename=doc.filename,
                    text=text,
                    page=i + 1,
                    section="fixture",
                    ordinal=i,
                    start=0,
                    end=len(text),
                )
            )
        repo.add_document(doc, chunks, "synthetic.txt")
        results = {}
        for label, scopes in [("uncached", 0), ("cached", 2)]:
            settings.search_cache_scopes = scopes
            search = HybridSearch(repo, NoEmbeddings(), settings)
            latencies, ranked = [], []
            for i in range(21):
                start = time.perf_counter()
                rows, _ = await search.search([f"topic{i % 7} retrieval"], [], 8, "lexical")
                latencies.append((time.perf_counter() - start) * 1000)
                ranked.append([c.id for c in rows])
            results[label] = {
                "cold_ms": latencies[0],
                "warm_p50_ms": float(np.median(latencies[1:])),
                "warm_p95_ms": float(np.percentile(latencies[1:], 95)),
                "cache": search.diagnostics(),
                "ranked_ids": ranked,
            }
        return {
            "kind": "synthetic_search_cache_timing",
            "chunks": count,
            "queries": 21,
            "rankings_identical": results["cached"]["ranked_ids"]
            == results["uncached"]["ranked_ids"],
            "results": {
                k: {x: y for x, y in v.items() if x != "ranked_ids"} for k, v in results.items()
            },
            "notes": (
                "BM25 CPU fixture in the build environment; "
                "excludes OCR, embeddings and LLM latency."
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("evaluation/search-cache.json"))
    args = parser.parse_args()
    if not 10 <= args.chunks <= 20000:
        parser.error("--chunks must be 10–20000")
    result = asyncio.run(measure(args.chunks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
