"""Real pinned-SWIFT CPU check with a tiny local Qwen; no 4B download and no quantization."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MODELSCOPE_CACHE"] = str(ROOT / ".tmp/qr-cache")


def main():
    print("Importing pinned SWIFT/Transformers...", flush=True)
    import torch
    from swift.llm import sft_main
    from swift.plugin import extra_callbacks
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from notebooks.kaggle_qwen_reranker_training import (
        reranker_arguments,
        reranker_row,
        set_reranker_environment,
        verify_swift_prompt,
    )
    from notebooks.kaggle_qwen_training import complete_checkpoints, deadline_callback
    from notebooks.qwen_reranker_scorer import DEFAULT_INSTRUCTION, QwenRerankerScorer

    root = Path(tempfile.mkdtemp(prefix="qr-smoke-"))
    base = root / "base"
    words = [
        "[UNK]",
        "[PAD]",
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<think>",
        "</think>",
        "yes",
        "no",
        "system",
        "user",
        "assistant",
        "Instruct",
        "Query",
        "Document",
        ":",
        "why",
        "evidence",
        "research",
        "positive",
        "negative",
        "cooking",
        "retrieval",
    ]
    vocab = {text: i for i, text in enumerate(words)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        additional_special_tokens=words[2:7],
    )
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=512,
            pad_token_id=vocab["[PAD]"],
            eos_token_id=vocab["<|im_end|>"],
            tie_word_embeddings=True,
        )
    )
    model.save_pretrained(base)
    fast.save_pretrained(base)
    row = reranker_row(
        "why evidence", "positive research evidence", "negative cooking", DEFAULT_INSTRUCTION
    )
    set_reranker_environment()
    prompt = verify_swift_prompt(base, row, DEFAULT_INSTRUCTION, 256)
    print("Prompt check:", prompt, flush=True)
    for split in ("train", "validation"):
        (root / f"{split}.jsonl").write_text((json.dumps(row) + "\n") * 16, encoding="utf-8")
    config = {
        "base_path": str(base),
        "train_data": str(root / "train.jsonl"),
        "validation_data": str(root / "validation.jsonl"),
        "max_length": 256,
        "learning_rate": 1e-3,
        "max_steps": 2,
        "output_dir": str(root / "checkpoints"),
    }
    args = reranker_arguments(config)
    options = dict(zip(args[::2], args[1::2], strict=True))
    options = {
        key: value for key, value in options.items() if not key.startswith(("--quant_", "--bnb_"))
    }
    options.update(
        {
            "--torch_dtype": "float32",
            "--fp16": "false",
            "--bf16": "false",
            "--eval_steps": "1",
            "--save_steps": "1",
            "--lora_rank": "4",
            "--gradient_checkpointing": "false",
            "--gradient_accumulation_steps": "1",
        }
    )
    extra_callbacks.append(deadline_callback(time.time() + 300, root / "progress.json"))
    result = sft_main([item for pair in options.items() for item in pair])
    assert result["global_step"] == 2
    assert json.loads((root / "training_integrity.json").read_text())["adapter_updated"]
    selected = complete_checkpoints(config["output_dir"])[-1][1]
    scorer = QwenRerankerScorer(base, selected, device="cpu", quantized=False, max_length=256)
    scores = scorer.score(
        [("why evidence", "positive research evidence"), ("why evidence", "negative cooking")],
        batch_size=2,
    )
    assert len(scores) == 2 and all(torch.isfinite(torch.tensor(scores)))
    print(
        json.dumps(
            {
                "passed": True,
                "optimizer_steps": 2,
                "prompt_verified": True,
                "adapter_reload": True,
                "device": "cpu",
                "quantization_tested": False,
                "artifacts": str(root),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
