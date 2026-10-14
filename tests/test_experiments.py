import json
from pathlib import Path

import pytest

from research_assist.experiments.data import load_split, prepare_qasper, records, validated_pairs
from research_assist.experiments.evaluate import Variant, evaluate
from research_assist.experiments.metrics import retrieval_metrics

FIXTURE = Path(__file__).parent / "fixtures" / "qasper"


def test_prepare_gold_split_and_exclusions(tmp_path: Path) -> None:
    manifest = prepare_qasper(FIXTURE, tmp_path)
    assert manifest["statistics"]["train_eligible"] == 4
    assert manifest["statistics"]["test_unanswerable"] == 1
    assert manifest["statistics"]["test_non_text_evidence"] == 1
    corpus, queries = load_split(tmp_path, "test")
    assert len(queries) == 2
    assert all(c["split"] == "test" for c in corpus)
    assert all(p["paper_id"].startswith("train") for p in validated_pairs(tmp_path))


def test_paper_leakage_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for split in ("train", "validation", "test"):
        (raw / f"{split}.json").write_bytes((FIXTURE / "train.json").read_bytes())
    with pytest.raises(ValueError, match="leakage"):
        prepare_qasper(raw, tmp_path / "prepared")


def test_synthetic_requires_review_and_train_provenance(tmp_path: Path) -> None:
    prepare_qasper(FIXTURE, tmp_path)
    synthetic = tmp_path / "synthetic.jsonl"
    synthetic.write_text(
        json.dumps(
            {
                "question": "Synthetic query",
                "positive_id": "test-0:1",
                "paper_id": "test-0",
                "reviewed": False,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="reviewed"):
        validated_pairs(tmp_path, synthetic)
    synthetic.write_text(synthetic.read_text().replace("false", "true"))
    with pytest.raises(ValueError, match="leaks"):
        validated_pairs(tmp_path, synthetic)


def test_hf_columnar_records() -> None:
    assert records({"section_name": ["A"], "paragraphs": [["Text"]]}) == [
        {"section_name": "A", "paragraphs": ["Text"]}
    ]


def test_metrics_multi_gold_and_duplicate_ranks() -> None:
    metrics = retrieval_metrics(["x", "a", "a", "b"], {"a", "b", "c"}, 3)
    assert metrics["recall@3"] == 2 / 3
    assert metrics["mrr@3"] == 0.5
    assert metrics["hit@3"] == 1
    assert 0 < metrics["ndcg@3"] < 1
    with pytest.raises(ValueError):
        retrieval_metrics([], set(), 10)


async def test_evaluation_produces_real_fixture_metrics(tmp_path: Path) -> None:
    prepare_qasper(FIXTURE, tmp_path)
    result = await evaluate(tmp_path, [Variant(name="test fixture", mode="bm25")])
    assert result["eligible_queries"] == 2
    assert result["kind"] == "retrieval_benchmark"
    assert result["variants"][0]["metrics"]["recall@10"] == 1
    assert result["variants"][0]["latency_p95_ms"] >= 0
