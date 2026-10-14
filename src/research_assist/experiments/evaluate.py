"""Same held-out paragraph corpus for BM25, dense, RRF and optional reranking."""

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from research_assist.experiments.data import file_hash, load_split
from research_assist.experiments.metrics import retrieval_metrics
from research_assist.infrastructure.embeddings import LocalEmbeddings
from research_assist.infrastructure.search import BM25Index


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    mode: Literal["bm25", "dense", "hybrid"]
    model: str = "intfloat/multilingual-e5-small"
    family: Literal["e5", "generic", "qwen3", "auto"] = "e5"
    revision: str = ""
    reranker: str = ""
    candidates: int = Field(default=20, ge=10, le=100)


async def evaluate(
    data: Path,
    variants: list[Variant],
    *,
    split: str = "test",
    device: str = "cpu",
    max_seq_length: int = 512,
) -> dict[str, object]:
    corpus, queries = load_split(data, split)
    output: dict[str, object] = {
        "format_version": 2,
        "kind": "retrieval_benchmark",
        "split": split,
        "scope": "all paragraphs in held-out split",
        "corpus_sha256": file_hash(data / "corpus.jsonl"),
        "queries_sha256": file_hash(data / f"{split}_queries.jsonl"),
        "eligible_queries": len(queries),
        "corpus_size": len(corpus),
        "device": device,
        "max_seq_length": max_seq_length,
    }
    results: list[dict[str, object]] = []
    texts, ids = [c["text"] for c in corpus], [c["id"] for c in corpus]
    for variant in variants:
        begin = time.perf_counter()
        lexical_index = BM25Index(texts) if variant.mode != "dense" else None
        vectors = None
        embedder = None
        reranker = None
        if variant.mode != "bm25":
            embedder = LocalEmbeddings(
                variant.model,
                32,
                family=variant.family,
                device=device,
                revision=variant.revision,
                max_seq_length=max_seq_length,
            )
            vectors = np.asarray(await embedder.embed(texts), dtype=np.float32)
        if variant.reranker:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder(variant.reranker, device=device, trust_remote_code=False)
            reranker.predict([("warmup", "warmup")])
        index_ms = (time.perf_counter() - begin) * 1000
        scores: dict[str, list[float]] = defaultdict(list)
        latencies, per_query = [], []
        for query in queries:
            start = time.perf_counter()
            lexical = lexical_index.score(query["question"]) if lexical_index else []
            dense = []
            if embedder is not None and vectors is not None:
                qvector = (await embedder.embed([query["question"]], query=True))[0]
                dense = (vectors @ np.asarray(qvector, dtype=np.float32)).tolist()
            if variant.mode == "hybrid":
                fused = [0.0] * len(ids)
                for values in (lexical, dense):
                    order = sorted(range(len(ids)), key=lambda i: (-values[i], ids[i]))[:100]
                    for rank, i in enumerate(order, 1):
                        if values is dense or values[i] > 0:
                            fused[i] += 1 / (60 + rank)
                values = fused
            else:
                values = lexical or dense
            order = sorted(range(len(ids)), key=lambda i: (-values[i], ids[i]))[
                : variant.candidates
            ]
            if variant.mode == "bm25":
                order = [i for i in order if values[i] > 0]
            if reranker:
                rescored = (
                    reranker.predict([(query["question"], texts[i]) for i in order]).tolist()
                    if order
                    else []
                )
                order = [
                    i
                    for _, i in sorted(
                        zip(rescored, order, strict=True), key=lambda p: (-p[0], ids[p[1]])
                    )
                ]
            elapsed = (time.perf_counter() - start) * 1000
            ranked = [ids[i] for i in order]
            row = {}
            for k in (5, 10):
                row.update(retrieval_metrics(ranked, set(query["relevant_ids"]), k))
            for key, value in row.items():
                scores[key].append(value)
            latencies.append(elapsed)
            per_query.append(
                {
                    "id": query["id"],
                    "ranked_ids": ranked[:10],
                    "metrics": row,
                    "retrieval_ms": elapsed,
                }
            )
        results.append(
            {
                "name": variant.name,
                "configuration": variant.model_dump(),
                "fingerprint": embedder.fingerprint if embedder else None,
                "metrics": {key: float(np.mean(values)) for key, values in scores.items()},
                "index_ms": index_ms,
                "latency_p50_ms": float(np.percentile(latencies, 50)),
                "latency_p95_ms": float(np.percentile(latencies, 95)),
                "per_query": per_query,
            }
        )
    output["variants"] = results
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--variants", type=Path, default=Path("configs/benchmark.json"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output", type=Path, default=Path("evaluation/retrieval.json"))
    args = parser.parse_args()
    variants = [Variant.model_validate(v) for v in json.loads(args.variants.read_text())]
    if len({v.name for v in variants}) != len(variants) or not variants:
        parser.error("Variant names must be unique and nonempty")
    result = asyncio.run(evaluate(args.data, variants, split=args.split, device=args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}; retrieval metrics are not end-to-end answer accuracy.")


if __name__ == "__main__":
    main()
