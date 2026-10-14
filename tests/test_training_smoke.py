"""Real optimization/save/reload on a tiny RANDOM model; not pretrained E5 quality."""

from pathlib import Path

import numpy as np
import pytest

from research_assist.experiments.data import prepare_qasper
from research_assist.experiments.train import TrainConfig, train
from research_assist.infrastructure.embeddings import LocalEmbeddings, checkpoint_signature


async def test_real_cpu_training_and_checkpoint_reload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("datasets")
    from sentence_transformers import SentenceTransformer, models
    from transformers import BertConfig, BertModel, BertTokenizerFast

    torch.set_num_threads(2)
    base = tmp_path / "random-base"
    base.mkdir()
    vocabulary = [
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
        "3",
    ]
    (base / "vocab.txt").write_text("\n".join(vocabulary))
    tokenizer = BertTokenizerFast(vocab_file=str(base / "vocab.txt"))
    tokenizer.save_pretrained(base)
    BertModel(
        BertConfig(
            vocab_size=len(vocabulary),
            hidden_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=128,
        )
    ).save_pretrained(base)
    transformer = models.Transformer(str(base), max_seq_length=64)
    model = SentenceTransformer(modules=[transformer, models.Pooling(24)], device="cpu")
    model.save_pretrained(base)
    initial = [p.detach().clone() for p in model.parameters()]
    before = checkpoint_signature(base)
    data = tmp_path / "data"
    prepare_qasper(Path(__file__).parent / "fixtures" / "qasper", data)
    output = tmp_path / "trained"
    result = train(
        TrainConfig(
            data=str(data),
            output=str(output),
            model=str(base),
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            max_seq_length=64,
            device="cpu",
        )
    )
    assert result["pairs_used"] == 4
    assert result["training_metrics"]["train_loss"] > 0
    assert checkpoint_signature(output) != before
    reloaded = LocalEmbeddings(str(output), 2, device="cpu", max_seq_length=64)
    assert reloaded.family == "e5"
    vectors = await reloaded.embed(["scientific questions"], query=True)
    assert len(vectors[0]) == 24
    assert np.isclose(np.linalg.norm(vectors[0]), 1)
    loaded = SentenceTransformer(str(output), device="cpu")
    assert any(not torch.allclose(a, b) for a, b in zip(initial, loaded.parameters(), strict=True))
