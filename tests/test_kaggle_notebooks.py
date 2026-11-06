"""Check the actual standalone cells, split safety, ranking metrics and exports offline."""

import ast
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("01_kaggle_finetune_e5.ipynb", "02_kaggle_finetune_reranker.ipynb")


def notebook_namespace(name=NAMES[1]):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    module = types.ModuleType("kaggle_notebook_smoke")
    module._RESEARCHDESK_ENV_READY = True
    sys.modules[module.__name__] = module
    for value in notebook["cells"]:
        source = "".join(value["source"])
        if value["cell_type"] == "code" and (
            "def prepare_qasper(" in source or "def export_checkpoint(" in source
        ):
            exec(compile(source, name, "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("name", NAMES)
def test_notebook_cells_are_standalone_and_unexecuted(name):
    book = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    assert (book["nbformat"], book["nbformat_minor"]) == (4, 5)
    ids = [cell["id"] for cell in book["cells"]]
    assert len(ids) == len(set(ids))
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            assert "from research_assist" not in source
            assert cell["execution_count"] is None and cell["outputs"] == []


def test_weak_negative_provenance_and_reviewed_mode(tmp_path):
    m = notebook_namespace()
    data = tmp_path / "data"
    m.prepare_qasper(ROOT / "tests/fixtures/qasper", data)
    corpus = {c["id"]: c for c in m.read_jsonl(data / "corpus.jsonl")}
    rows, review = m.make_reranker_pairs(data, negatives_per_query=2)
    assert {r["label"] for r in rows} == {0.0, 1.0}
    assert all(corpus[r["passage_id"]]["split"] == "train" for r in rows)
    assert all(not row["reviewed"] for row in review)
    queries = {q["id"]: q for q in m.read_jsonl(data / "train_queries.jsonl")}
    for row in rows:
        if row["label"] == 0:
            assert row["source"] == "cross_paper_weak"
            assert corpus[row["passage_id"]]["paper_id"] != queries[row["query_id"]]["paper_id"]
    path = tmp_path / "reviewed.jsonl"
    m.write_jsonl(path, [{**row, "reviewed": True} for row in review])
    reviewed, _ = m.make_reranker_pairs(data, reviewed_path=path)
    assert {r["source"] for r in reviewed if r["label"] == 0} == {"human_reviewed"}
    heldout_id = next(cid for cid, c in corpus.items() if c["split"] == "test")
    m.write_jsonl(path, [{**review[0], "negative_id": heldout_id, "reviewed": True}])
    with pytest.raises(ValueError, match="held-out"):
        m.make_reranker_pairs(data, reviewed_path=path)
    qid = review[0]["query_id"]
    m.write_jsonl(
        path, [{**review[0], "negative_id": queries[qid]["relevant_ids"][0], "reviewed": True}]
    )
    with pytest.raises(ValueError, match="known positive"):
        m.make_reranker_pairs(data, reviewed_path=path)


def test_metrics_penalize_missing_gold_without_injection():
    m = notebook_namespace()
    samples = [
        {
            "query_id": "found",
            "query": "question",
            "relevant_ids": ["gold"],
            "documents": [{"id": "gold", "text": "answer"}],
        },
        {
            "query_id": "missing",
            "query": "question",
            "relevant_ids": ["absent"],
            "documents": [{"id": "distractor", "text": "unrelated"}],
        },
        {"query_id": "empty", "query": "question", "relevant_ids": ["other"], "documents": []},
    ]
    metrics = m.evaluate_candidates(None, samples)
    assert metrics["queries"] == 3 and metrics["gold_injection"] is False
    assert metrics["metrics"]["ndcg@10"] == pytest.approx(1 / 3)
    assert metrics["metrics"]["candidate_recall"] == pytest.approx(1 / 3)
    assert m.rank_metrics(["gold", "gold"], {"gold"})["recall@10"] == 1


def test_zip_root_checksums_and_repeatable_export(tmp_path):
    m = notebook_namespace()
    model, experiment = tmp_path / "model", tmp_path / "experiment"
    model.mkdir()
    experiment.mkdir()
    for name in ("config.json", "research_model.json", "tokenizer_config.json"):
        (model / name).write_text("{}")
    (model / "model.safetensors").write_bytes(b"test fixture, not real weights")
    (experiment / "metrics.json").write_text("{}")
    (model / "checkpoints").mkdir()
    (model / "checkpoints/optimizer.pt").write_bytes(b"excluded")
    archive = tmp_path / "checkpoint.zip"
    for _ in range(2):
        m.export_checkpoint(model, archive, experiment, tmp_path / "experiment.zip")
        with zipfile.ZipFile(archive) as z:
            assert "config.json" in z.namelist()
            assert not any("checkpoints" in p for p in z.namelist())
            manifest = json.loads(z.read("artifact_manifest.json"))
            for name, digest in manifest["sha256"].items():
                assert m.hashlib.sha256(z.read(name)).hexdigest() == digest
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        m.unzip_checkpoint(bad, tmp_path / "bad-output")


@pytest.mark.parametrize("expired", [False, True])
def test_reranker_real_optimization_and_cpu_reload(tmp_path, monkeypatch, expired):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import CrossEncoder
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    torch.set_num_threads(2)
    m = notebook_namespace()
    monkeypatch.setattr(m, "package_versions", lambda: {"torch": torch.__version__})
    base = tmp_path / "base"
    base.mkdir()
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "query", "answer", "wrong"]
    (base / "vocab.txt").write_text("\n".join(words))
    BertTokenizerFast(vocab_file=str(base / "vocab.txt")).save_pretrained(base)
    initial = BertForSequenceClassification(
        BertConfig(
            vocab_size=len(words),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=24,
            num_labels=1,
            max_position_embeddings=64,
        )
    )
    initial.save_pretrained(base)
    before = {name: p.detach().clone() for name, p in initial.named_parameters()}
    rows = [
        {"query": "query", "passage": "answer", "label": 1.0, "source": "reviewed_positive"},
        {"query": "query", "passage": "wrong", "label": 0.0, "source": "human_reviewed"},
    ] * 3
    samples = [
        {
            "query_id": "val",
            "query": "query",
            "relevant_ids": ["gold"],
            "documents": [{"id": "gold", "text": "answer"}, {"id": "negative", "text": "wrong"}],
        }
    ]
    output = tmp_path / "trained"
    budget = m.NotebookTimeBudget(started=0, clock=lambda: 12 * 3600) if expired else None
    result = m.train_reranker(
        str(base),
        output,
        rows,
        samples,
        epochs=1,
        batch_size=2,
        accumulation=2,
        learning_rate=1e-3,
        max_length=32,
        device="cpu",
        time_budget=budget,
    )
    loaded = CrossEncoder(str(output), device="cpu", local_files_only=True)
    if expired:
        assert result["optimizer_steps"] == 0 and result["fine_tuned"] is False
        assert result["selection"] == "base_not_trained"
        assert result["validation_after"] is None and result["stop_reason"] == "time_budget"
    else:
        assert result["best_epoch"] == 1 and result["history"][0]["train_loss"] > 0
        assert any(
            not torch.allclose(before[name], p) for name, p in loaded.model.named_parameters()
        )
    assert len(loaded.predict([("query", "answer")])) == 1


@pytest.mark.parametrize("stop_hour", [None, 10, 11])
def test_e5_notebook_cached_loss_optimization_and_reload(tmp_path, monkeypatch, stop_hour):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("datasets")
    from sentence_transformers import SentenceTransformer, models
    from transformers import BertConfig, BertModel, BertTokenizerFast

    torch.set_num_threads(2)
    m = notebook_namespace(NAMES[0])
    book = json.loads((ROOT / "notebooks" / NAMES[0]).read_text(encoding="utf-8"))
    source = next(
        "".join(c["source"]) for c in book["cells"] if "class TrainConfig:" in "".join(c["source"])
    )
    assert "CachedMultipleNegativesRankingLoss" in source
    exec(compile(source, NAMES[0], "exec"), m.__dict__)
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    words = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "query",
        "passage",
        ":",
        "the",
        "study",
        "uses",
        "retrieval",
        "evidence",
        "language",
        "models",
        "scientific",
        "questions",
        "train",
        "validation",
        "test",
        "0",
        "1",
        "2",
    ]
    (transformer / "vocab.txt").write_text("\n".join(words))
    BertTokenizerFast(vocab_file=str(transformer / "vocab.txt")).save_pretrained(transformer)
    BertModel(
        BertConfig(
            vocab_size=len(words),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=24,
            max_position_embeddings=128,
        )
    ).save_pretrained(transformer)
    model = SentenceTransformer(
        modules=[
            models.Transformer(str(transformer), max_seq_length=64),
            models.Pooling(16),
        ],
        device="cpu",
    )
    base = tmp_path / "base"
    model.save_pretrained(base)
    before = [p.detach().clone() for p in model.parameters()]
    data = tmp_path / "data"
    m.prepare_qasper(ROOT / "tests/fixtures/qasper", data)
    output = tmp_path / "trained"
    now = [0]
    budget = m.NotebookTimeBudget(started=0, clock=lambda: now[0]) if stop_hour else None
    if stop_hour:
        factory = m.make_time_budget_callback

        def stop_after_first_step(budget, minutes):
            callback = factory(budget, minutes)
            original_step = callback.on_step_end

            def step(*args, **kwargs):
                # The first optimizer step can have LR=0 during warmup. Let one
                # effective update run before testing deadline/export behavior.
                if args[1].global_step >= 2:
                    now[0] = stop_hour * 3600
                return original_step(*args, **kwargs)

            callback.on_step_end = step
            return callback

        monkeypatch.setattr(m, "make_time_budget_callback", stop_after_first_step)
    manifest = m.train(
        m.TrainConfig(
            data=str(data),
            output=str(output),
            model=str(base),
            epochs=2 if stop_hour else 1,
            batch_size=2,
            mini_batch_size=1,
            learning_rate=1e-3,
            max_seq_length=64,
            device="cpu",
        ),
        time_budget=budget,
    )
    loaded = SentenceTransformer(str(output), device="cpu", local_files_only=True)
    assert manifest["loss"] == "CachedMultipleNegativesRankingLoss"
    assert manifest["pairs_used"] == 4
    assert any(not torch.allclose(a, b) for a, b in zip(before, loaded.parameters(), strict=True))
    assert loaded.encode(["query: scientific questions"]).shape == (1, 16)
    if stop_hour:
        assert manifest["optimizer_steps"] == 2 and manifest["stop_reason"] == "time_budget"
        assert manifest["fine_tuned"] is True
    if stop_hour == 11:
        assert manifest["validation_after"] is None
        assert manifest["selection"] == "last_unvalidated"


def test_session_budget_keeps_export_reserve():
    m = notebook_namespace()
    now = [0]
    budget = m.NotebookTimeBudget(started=0, clock=lambda: now[0])
    assert not budget.training_expired() and not budget.evaluation_expired()
    now[0] = 10 * 3600
    assert budget.training_expired() and not budget.evaluation_expired()
    now[0] = 11 * 3600
    assert budget.evaluation_expired()
    assert budget.snapshot()["export_reserve_hours"] == 1
    with pytest.raises(ValueError, match="Require"):
        m.NotebookTimeBudget(train_stop_hours=12)


def test_evaluation_deadline_discards_partial_metrics_and_removes_hook():
    torch = pytest.importorskip("torch")
    m = notebook_namespace()
    now = [0]
    budget = m.NotebookTimeBudget(started=0, clock=lambda: now[0])
    model = types.SimpleNamespace(model=torch.nn.Linear(1, 1))

    def incomplete_evaluation():
        model.model(torch.ones(1, 1))
        now[0] = 11 * 3600
        model.model(torch.ones(1, 1))
        return {"incorrect_partial_metric": 1.0}

    with pytest.raises(m.NotebookDeadlineReached, match="batch boundary"):
        m.evaluate_with_budget(model, incomplete_evaluation, budget)
    assert not model.model._forward_pre_hooks


def test_trainer_callback_requests_periodic_and_deadline_checkpoint():
    pytest.importorskip("transformers")
    m = notebook_namespace()
    now = [0]
    budget = m.NotebookTimeBudget(started=0, clock=lambda: now[0])
    callback = m.make_time_budget_callback(budget, checkpoint_minutes=15)
    callback.last_save -= 901
    control = types.SimpleNamespace(
        should_save=False, should_training_stop=False, should_evaluate=True
    )
    callback.on_step_end(None, None, control)
    assert control.should_save and not control.should_training_stop
    now[0] = 10 * 3600
    control.should_save = False
    callback.on_step_end(None, None, control)
    assert control.should_training_stop and control.should_save and not control.should_evaluate
