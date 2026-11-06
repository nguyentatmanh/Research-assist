"""Reranker notebook semantics: full-pair prompts, role labels, leakage and ranking metrics."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from notebooks.kaggle_qwen_reranker_training import (
    export_reranker_adapter,
    prepare_reranker_data,
    ranking_metrics,
    reranker_arguments,
    set_reranker_environment,
)
from notebooks.qwen_adapter_encoder import read_adapter_manifest
from notebooks.qwen_reranker_scorer import (
    DEFAULT_INSTRUCTION,
    PREFIX,
    SUFFIX,
    QwenRerankerScorer,
    pair_token_ids,
    read_reranker_manifest,
)
from research_assist.experiments.data import write_jsonl

ROOT = Path(__file__).parents[1]


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


@pytest.fixture
def data(tmp_path):
    corpus, pairs = [], []
    for split, size in (("train", 20), ("validation", 6), ("test", 1)):
        queries = []
        for i in range(size):
            paper = f"{split}-{i}"
            positive, other_gold, negative = paper + ":p", paper + ":g", paper + ":n"
            for cid, text in (
                (positive, f"gold research evidence {paper}"),
                (other_gold, f"another research gold {paper}"),
                (negative, f"research topic unrelated {paper}"),
            ):
                corpus.append({"id": cid, "paper_id": paper, "split": split, "text": text})
            query = {
                "id": paper,
                "paper_id": paper,
                "question": f"research evidence {paper}",
                "relevant_ids": [positive, other_gold],
                "excluded_reason": "",
            }
            queries.append(query)
            if split == "train":
                pairs.append(
                    {
                        "query_id": paper,
                        "question": query["question"],
                        "paper_id": paper,
                        "positive_id": positive,
                        "reviewed": True,
                    }
                )
        write_jsonl(tmp_path / f"{split}_queries.jsonl", queries)
    write_jsonl(tmp_path / "corpus.jsonl", corpus)
    write_jsonl(tmp_path / "train_pairs.jsonl", pairs)
    return tmp_path


def test_standalone_notebook_uses_reranker_workers_without_starting_them():
    book = json.loads(
        (ROOT / "notebooks/07_kaggle_finetune_qwen3_reranker_4b.ipynb").read_text("utf-8")
    )
    assert len({cell["id"] for cell in book["cells"]}) == len(book["cells"])
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and cell["outputs"] == []
    scope = {"__name__": "__main__"}
    exec(compile("".join(book["cells"][2]["source"]), "reranker-library", "exec"), scope)
    worker = scope["QWEN_WORKER_SOURCE"]
    ast.parse(worker)
    assert "from notebooks." not in worker and "from research_assist." not in worker
    assert callable(scope["train_reranker_worker"]) and callable(scope["verify_swift_prompt"])
    source = "\n".join("".join(cell["source"]) for cell in book["cells"])
    assert 'BASE_MODEL = "Qwen/Qwen3-Reranker-4B"' in source
    for term in ("10 * 3600", "10.5 * 3600", "11 * 3600", "system_site_packages=True"):
        assert term in source
    assert "START_CLOUDFLARE" not in source


def test_args_use_pointwise_generative_bce_and_exact_two_pair_groups(monkeypatch):
    for name in (
        "MAX_POSITIVE_SAMPLES",
        "MAX_NEGATIVE_SAMPLES",
        "GENERATIVE_RERANKER_POSITIVE_TOKEN",
        "GENERATIVE_RERANKER_NEGATIVE_TOKEN",
    ):
        monkeypatch.setenv(name, "before-test")
    set_reranker_environment()
    import os

    assert os.environ["MAX_POSITIVE_SAMPLES"] == os.environ["MAX_NEGATIVE_SAMPLES"] == "1"
    assert os.environ["GENERATIVE_RERANKER_POSITIVE_TOKEN"] == "yes"
    config = {
        "base_path": "base",
        "train_data": "train",
        "validation_data": "val",
        "max_length": 512,
        "learning_rate": 6e-6,
        "max_steps": 500,
        "output_dir": "out",
    }
    args = reranker_arguments(config)
    values = dict(zip(args[::2], args[1::2], strict=True))
    assert values["--model_type"] == "qwen3_reranker"
    assert values["--task_type"] == values["--loss_type"] == "generative_reranker"
    assert values["--per_device_train_batch_size"] == "1"
    assert values["--quant_bits"] == "4" and values["--bf16"] == "false"
    assert values["--save_only_model"] == "false" and "infonce" not in args


def test_prompt_budget_counts_whole_pair_and_preserves_assistant_suffix():
    tokenizer = Tokenizer()
    overhead = len(PREFIX) + len(SUFFIX)
    with pytest.raises(ValueError, match="overhead"):
        pair_token_ids(tokenizer, "q", "d", "instruction", overhead)
    with pytest.raises(ValueError, match="truncation"):
        pair_token_ids(tokenizer, "query", "gold" * 100, "task", overhead + 60)
    ids = pair_token_ids(tokenizer, "query", "gold" * 100, "task", overhead + 60, truncate=True)
    assert len(ids) == overhead + 60
    assert ids[: len(PREFIX)] == tokenizer.encode(PREFIX)
    assert ids[-len(SUFFIX) :] == tokenizer.encode(SUFFIX)


def test_data_groups_use_raw_query_assistant_documents_and_exclude_all_gold(data):
    train, val, panel, stats = prepare_reranker_data(data, Tokenizer(), "task", 2000)
    assert len(train) == 20 and len(val) == 6
    assert stats["train_pairs"] == 40
    for row in train:
        assert row["messages"][0] == {"role": "system", "content": "task"}
        assert row["messages"][1]["role"] == "user"
        assert not row["messages"][1]["content"].startswith("Instruct:")
        assert row["positive_messages"][0][0]["role"] == "assistant"
        assert row["negative_messages"][0][0]["role"] == "assistant"
        assert "unrelated" in row["negative_messages"][0][0]["content"]
    assert all(item["split"] in {"train", "validation"} for item in stats["provenance"])
    assert all(
        candidate["split"] == "validation" for item in panel for candidate in item["candidates"]
    )


def test_data_rejects_paper_leakage_before_training(data):
    path = data / "corpus.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["paper_id"] = "train-0"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="paper leakage"):
        prepare_reranker_data(data, Tokenizer(), "task", 2000)


def test_training_drops_overlong_gold_without_silently_cutting_it(data):
    path = data / "corpus.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    next(row for row in rows if row["id"] == "train-0:p")["text"] = "gold" * 600
    write_jsonl(path, rows)
    train, _, _, stats = prepare_reranker_data(data, Tokenizer(), "task", 2000)
    assert len(train) == 19 and stats["excluded_train_groups"] == 1
    assert all("gold" * 600 not in row["positive_messages"][0][0]["content"] for row in train)


def test_evaluation_candidates_do_not_inject_gold(data, monkeypatch):
    from notebooks import kaggle_qwen_reranker_training as module

    class MissesGold:
        def __init__(self, texts):
            self.texts = texts

        def rank(self, query, count):
            return [i for i, text in enumerate(self.texts) if "unrelated" in text][:count]

    monkeypatch.setattr(module, "NotebookBM25", MissesGold)
    _, _, panel, _ = prepare_reranker_data(data, Tokenizer(), "task", 2000)
    assert all(
        not set(item["relevant_ids"]) & {c["id"] for c in item["candidates"]} for item in panel
    )
    assert ranking_metrics(panel)["candidate_recall@20"] == 0


def test_ranking_missing_gold_stays_zero_and_cannot_be_injected():
    panel = [{"relevant_ids": ["gold"], "candidates": [{"id": "other"}]}]
    metrics = ranking_metrics(panel, [[100]])
    assert metrics == {"ndcg@10": 0, "mrr@10": 0, "recall@10": 0, "candidate_recall@20": 0}
    panel = [{"relevant_ids": ["gold"], "candidates": [{"id": "other"}, {"id": "gold"}]}]
    assert ranking_metrics(panel)["mrr@10"] == 0.5
    assert ranking_metrics(panel, [[0, 1]])["ndcg@10"] == 1
    with pytest.raises(ValueError):
        ranking_metrics(panel, [[float("nan"), 1]])


def test_scorer_uses_yes_minus_no_not_embedding_pooling():
    torch = pytest.importorskip("torch")

    class Batch(dict):
        def to(self, device):
            return self

    class FakeTokenizer(Tokenizer):
        def pad(self, values, **kwargs):
            return Batch(input_ids=torch.tensor(values["input_ids"]))

    class Model:
        device = "cpu"

        def __call__(self, **kwargs):
            assert kwargs["logits_to_keep"] == 1 and kwargs["use_cache"] is False
            return SimpleNamespace(logits=torch.tensor([[[5.0, 2.0]]]))

    scorer = QwenRerankerScorer.__new__(QwenRerankerScorer)
    scorer.tokenizer, scorer.model = FakeTokenizer(), Model()
    scorer.instruction, scorer.max_length, scorer.yes, scorer.no = "task", 2000, 1, 0
    assert scorer.score([("q", "d")]) == [-3.0]


def test_export_keeps_reranker_identity_separate_from_embedding(tmp_path):
    checkpoint = tmp_path / "checkpoints/checkpoint-2"
    checkpoint.mkdir(parents=True)
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / name).write_text("test-only")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 2}))
    (tmp_path / "training_integrity.json").write_text(
        '{"adapter_updated": true, "optimizer_steps": 2}'
    )
    config = {
        "run": str(tmp_path),
        "output_dir": str(checkpoint.parent),
        "base_model": "Qwen/Qwen3-Reranker-4B",
        "base_revision": "a" * 40,
        "instruction": DEFAULT_INSTRUCTION,
        "max_length": 512,
        "learning_rate": 6e-6,
        "data_sha256": {},
        "recipe": "test",
    }
    output = export_reranker_adapter(config, {"timed_out": True})
    manifest = read_reranker_manifest(output)
    assert manifest["task_type"] == manifest["loss_type"] == "generative_reranker"
    assert manifest["optimizer_steps"] == 2
    assert (tmp_path / "resume/optimizer.pt").exists()
    with pytest.raises(ValueError):
        read_adapter_manifest(output)
