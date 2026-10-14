"""Binary relevance metrics at paragraph level; deduplicate rankings before scoring."""

import math


def retrieval_metrics(ranked: list[str], relevant: set[str], k: int) -> dict[str, float]:
    if not relevant or k < 1:
        raise ValueError("Metrics require nonempty gold and positive k")
    ranked = list(dict.fromkeys(ranked))[:k]
    hits = [i + 1 for i, cid in enumerate(ranked) if cid in relevant]
    dcg = sum(1 / math.log2(rank + 1) for rank in hits)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return {
        f"recall@{k}": len(hits) / len(relevant),
        f"hit@{k}": float(bool(hits)),
        f"mrr@{k}": 1 / hits[0] if hits else 0.0,
        f"ndcg@{k}": dcg / ideal,
    }
