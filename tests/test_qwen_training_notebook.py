"""Offline checks for the Qwen recipe, checkpoints, deadline behavior and inference policy."""

import ast
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from notebooks.kaggle_qwen_training import (
    complete_checkpoints,
    deadline_callback,
    export_adapter,
    make_triplets,
    run_bounded,
    session_start,
    swift_arguments,
)
from notebooks.qwen_adapter_encoder import (
    adapter_identity,
    extract_adapter_archive,
    read_adapter_manifest,
)

ROOT = Path(__file__).parents[1]


def test_notebook_standalone_library_does_not_start_worker(monkeypatch):
    notebook = json.loads((ROOT / "notebooks/06_kaggle_finetune_qwen3_4b.ipynb").read_text("utf-8"))
    scope = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and cell["outputs"] == []
    # Only load embedded stdlib definitions, not setup/network/training.
    source = "".join(notebook["cells"][2]["source"])
    exec(compile(source, "notebook-library", "exec"), scope)
    assert callable(scope["train_worker"]) and callable(scope["prepare_worker"])
    ast.parse(scope["QWEN_WORKER_SOURCE"])
    assert "from notebooks." not in scope["QWEN_WORKER_SOURCE"]
    assert "from research_assist." not in scope["QWEN_WORKER_SOURCE"]
    all_source = "\n".join("".join(c["source"]) for c in notebook["cells"])
    for term in ("10 * 3600", "10.5 * 3600", "11 * 3600", "system_site_packages=True"):
        assert term in all_source
    assert "START_CLOUDFLARE" not in all_source


def test_swift_embedding_recipe_has_negatives_and_t4_precision():
    config = {
        "base_path": "pinned",
        "train_data": "train",
        "validation_data": "val",
        "max_length": 512,
        "learning_rate": 2e-5,
        "max_steps": 500,
        "output_dir": "out",
    }
    args = swift_arguments(config)
    values = dict(zip(args[::2], args[1::2], strict=True))
    assert values["--model_type"] == "qwen3_emb"
    assert values["--task_type"] == "embedding" and values["--loss_type"] == "infonce"
    assert values["--quant_bits"] == "4" and values["--bnb_4bit_quant_type"] == "nf4"
    assert values["--torch_dtype"] == "float16" and values["--bf16"] == "false"
    assert values["--save_only_model"] == "false"
    assert values["--split_dataset_ratio"] == "0"


def test_callback_saves_at_boundary_and_verifies_actual_parameter_change(tmp_path):
    torch = pytest.importorskip("torch")

    model = torch.nn.Linear(2, 2)
    clock = [0]
    callback = deadline_callback(100, tmp_path / "progress.json", 15, clock=lambda: clock[0])
    state = SimpleNamespace(global_step=1)
    callback.on_train_begin(None, state, None, model=model)
    clock[0] = 16
    control = SimpleNamespace(should_training_stop=False, should_save=False, should_evaluate=True)
    assert callback.on_step_end(None, state, control).should_save
    callback.on_save(None, state, control, model=model)
    assert not json.loads((tmp_path / "training_integrity.json").read_text())["adapter_updated"]
    with torch.no_grad():
        model.weight.add_(0.1)
    callback.on_save(None, state, control, model=model)
    assert json.loads((tmp_path / "training_integrity.json").read_text())["adapter_updated"]
    clock[0] = 101
    callback.on_step_end(None, state, control)
    assert control.should_training_stop and control.should_save and not control.should_evaluate


def checkpoint(root, step, best=None):
    directory = root / f"checkpoint-{step}"
    directory.mkdir(parents=True)
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (directory / name).write_bytes(b"test-only")
    (directory / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": step,
                "best_model_checkpoint": str(best) if best else None,
            }
        )
    )
    return directory


def test_export_preserves_best_adapter_and_last_resumable_checkpoint(tmp_path):
    output = tmp_path / "checkpoints"
    best = checkpoint(output, 5)
    latest = checkpoint(output, 10, best)
    incomplete = checkpoint(output, 12)
    (incomplete / "optimizer.pt").unlink()
    assert [c[0] for c in complete_checkpoints(output)] == [5, 10]
    (tmp_path / "training_integrity.json").write_text(
        '{"adapter_updated": true, "optimizer_steps": 10}'
    )
    config = {
        "run": str(tmp_path),
        "output_dir": str(output),
        "base_model": "Qwen/Qwen3-Embedding-4B",
        "base_revision": "a" * 40,
        "instruction": "Find research passages",
        "max_length": 512,
        "data_sha256": {"train": "b" * 64},
        "learning_rate": 2e-5,
    }
    exported = export_adapter(config, {"timed_out": True})
    assert read_adapter_manifest(exported)["optimizer_steps"] == 5
    assert json.loads((tmp_path / "resume/trainer_state.json").read_text())["global_step"] == 10
    assert (tmp_path / "resume/optimizer.pt").read_bytes() == (latest / "optimizer.pt").read_bytes()
    first = adapter_identity(exported, 512)
    (exported / "adapter_model.safetensors").write_bytes(b"changed-weights")
    assert first != adapter_identity(exported, 512)


def test_no_checkpoint_or_no_actual_update_cannot_export_base(tmp_path):
    with pytest.raises(RuntimeError, match="weight update"):
        export_adapter({"run": str(tmp_path), "output_dir": str(tmp_path)}, {})
    (tmp_path / "research_model.json").write_text(json.dumps({"fine_tuned": False}))
    with pytest.raises(ValueError):
        read_adapter_manifest(tmp_path)


def test_adapter_zip_rejects_path_escape(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("../escaped.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_adapter_archive(archive, tmp_path / "output")
    assert not (tmp_path / "escaped.txt").exists()


def test_session_clock_does_not_reset_on_rerun(tmp_path):
    clock = tmp_path / "clock.json"
    first = session_start(clock, 2)
    second = session_start(clock, 0)
    assert second <= first


def test_supervisor_terminates_worker_when_callback_cannot_finish(tmp_path, monkeypatch):
    from notebooks import kaggle_qwen_training as module

    class Process:
        returncode = None
        terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, **kwargs):
            return self.returncode

    process = Process()
    clock = [0]
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: clock.__setitem__(0, 20))
    status = run_bounded(["fake-worker"], tmp_path / "train.log", 10)
    assert status["timed_out"] and process.terminated and status["returncode"] == -15


def test_triplets_respect_paper_gold_and_nontruncated_spans(tmp_path):
    from research_assist.experiments.data import write_jsonl

    corpus, pairs, train_queries, validation_queries = [], [], [], []
    for split, count in (("train", 20), ("validation", 5)):
        for i in range(count):
            paper = f"{split}-{i}"
            positive, negative = paper + ":p", paper + ":n"
            corpus += [
                {
                    "id": positive,
                    "paper_id": paper,
                    "split": split,
                    "text": f"gold evidence {paper}",
                },
                {
                    "id": negative,
                    "paper_id": paper,
                    "split": split,
                    "text": f"candidate negative {paper}",
                },
            ]
            q = {
                "id": paper,
                "question": f"evidence question {paper}",
                "paper_id": paper,
                "relevant_ids": [positive],
                "excluded_reason": "",
            }
            if split == "train":
                train_queries.append(q)
                pairs.append(
                    {
                        "question": q["question"],
                        "positive_id": positive,
                        "paper_id": paper,
                        "reviewed": True,
                    }
                )
            else:
                validation_queries.append(q)
    for name, rows in (
        ("corpus", corpus),
        ("train_pairs", pairs),
        ("train_queries", train_queries),
        ("validation_queries", validation_queries),
    ):
        write_jsonl(tmp_path / f"{name}.jsonl", rows)
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(range(len(text.split()))))
    train, val, stats = make_triplets(tmp_path, tokenizer, "Find passages")
    assert len(train) == 20 and len(val) == 5 and stats["excluded"] == 0
    for row in train:
        assert "Instruct: Find passages\nQuery:" in row["messages"][0]["content"]
        positive = row["positive_messages"][0][0]["content"]
        negative = row["negative_messages"][0][0]["content"]
        assert positive != negative and positive.split()[-1] == negative.split()[-1]


def test_adapter_query_eos_pooling_and_normalization_on_cpu():
    torch = pytest.importorskip("torch")

    from notebooks.qwen_adapter_encoder import QwenAdapterEncoder

    seen = []

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            seen.extend(texts)
            return {"input_ids": [[1, 2] for _ in texts]}

        def pad(self, tokens, **kwargs):
            ids = torch.tensor(tokens["input_ids"])
            assert ids.tolist() == [[1, 2, 9]]
            return Batch(input_ids=ids, attention_mask=torch.ones_like(ids))

    class Batch(dict):
        def to(self, device):
            return self

    class Transformer:
        def __call__(self, **kwargs):
            return SimpleNamespace(
                last_hidden_state=torch.tensor([[[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]]])
            )

    encoder = QwenAdapterEncoder.__new__(QwenAdapterEncoder)
    encoder.instruction, encoder.max_seq_length, encoder.suffix = "Find passages", 512, 9
    encoder.tokenizer = Tokenizer()
    encoder.model = SimpleNamespace(model=Transformer(), device="cpu")
    result = encoder.encode_query(["why?"])
    assert result[0].tolist() == pytest.approx([0.6, 0.8])
    assert seen == ["Instruct: Find passages\nQuery:why?"]
    seen.clear()
    encoder.encode_document(["a passage"])
    assert seen == ["a passage"]
