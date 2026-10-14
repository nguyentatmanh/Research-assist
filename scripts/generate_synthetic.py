"""Generate TRAIN-only draft questions. Human review is required before fine-tuning."""

import argparse
import asyncio
import json
import random
from pathlib import Path

import httpx
from pydantic import Field

from research_assist.config import Settings
from research_assist.domain.models import Contract
from research_assist.experiments.data import normalize, read_jsonl, write_jsonl
from research_assist.infrastructure.llm import HTTPLanguageModel, usage_scope
from research_assist.infrastructure.routing import RequestGate


class SyntheticQuestion(Contract):
    question: str = Field(min_length=8, max_length=1000)
    supporting_quote: str = Field(min_length=20, max_length=1400)


async def generate(data: Path, output: Path, limit: int, language: str, seed: int) -> None:
    settings = Settings()
    if settings.llm_provider == "demo":
        raise ValueError("Choose a real provider and configure its key in .env")
    if await asyncio.to_thread(output.exists):
        raise ValueError("Choose a new output file to preserve previous reviews")
    corpus = read_jsonl(data / "corpus.jsonl")
    heldout = {normalize(c["text"]) for c in corpus if c["split"] != "train"}
    candidates = [
        c
        for c in corpus
        if c["split"] == "train"
        and 100 <= len(c["text"]) <= 3000
        and normalize(c["text"]) not in heldout
    ]
    random.Random(seed).shuffle(candidates)
    results = []
    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        llm = HTTPLanguageModel(settings, client, RequestGate(settings.llm_requests_per_minute))
        with usage_scope(cache_enabled=False) as usage:
            for paragraph in candidates[:limit]:
                item = await llm.generate(
                    task="synthetic_question",
                    instruction=(
                        f"Create one information-seeking research question in {language}, "
                        "answered by this passage only. Copy a supporting quote verbatim. "
                        "Do not include the answer in the question."
                    ),
                    payload={"passage": paragraph["text"]},
                    schema=SyntheticQuestion,
                    fallback=lambda: SyntheticQuestion(
                        question="No demo", supporting_quote="No extractive demo available"
                    ),
                )
                if normalize(item.supporting_quote) not in normalize(paragraph["text"]):
                    continue
                results.append(
                    {
                        "query_id": f"synthetic-{paragraph['id']}",
                        "question": item.question,
                        "positive_id": paragraph["id"],
                        "paper_id": paragraph["paper_id"],
                        "supporting_quote": item.supporting_quote,
                        "source": "synthetic",
                        "reviewed": False,
                        "generator": llm.model,
                        "seed": seed,
                    }
                )
                # Checkpoint every accepted draft; interrupted calls do not erase work.
                write_jsonl(output, results)
            print(
                json.dumps(
                    {
                        "drafts": len(results),
                        "requests": usage.calls,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "output": str(output),
                    }
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/experiments/qasper"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/vi_drafts.jsonl"))
    parser.add_argument("--limit", type=int, choices=range(1, 21), default=8)
    parser.add_argument("--language", choices=["Vietnamese", "English"], default="Vietnamese")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(generate(args.data, args.output, args.limit, args.language, args.seed))
