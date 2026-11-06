"""Optional real SWIFT CPU smoke check in an isolated environment, without model downloads."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MODELSCOPE_CACHE"] = str(Path(__file__).resolve().parents[1] / ".tmp/qs-cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import faulthandler

    faulthandler.enable()
    faulthandler.dump_traceback_later(120)
    print("Importing pinned Transformers...", flush=True)
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from notebooks.kaggle_qwen_training import (
        complete_checkpoints,
        deadline_callback,
        swift_arguments,
    )

    print("Building tiny local Qwen fixture...", flush=True)

    root = Path(tempfile.mkdtemp(prefix="qwen-swift-smoke-"))
    base = root / "base"
    vocab = {
        text: i
        for i, text in enumerate(
            [
                "[UNK]",
                "[PAD]",
                "<|endoftext|>",
                "Instruct",
                ":",
                "Find",
                "passages",
                "Query",
                "why",
                "evidence",
                "research",
                "positive",
                "negative",
                "cooking",
                "retrieval",
            ]
        )
    }
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="<|endoftext|>"
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
            max_position_embeddings=128,
            pad_token_id=vocab["[PAD]"],
            eos_token_id=vocab["<|endoftext|>"],
            tie_word_embeddings=True,
        )
    )
    model.save_pretrained(base)
    fast.save_pretrained(base)
    row = {
        "messages": [{"role": "user", "content": "Instruct: Find passages\nQuery:why evidence"}],
        "positive_messages": [[{"role": "user", "content": "positive research evidence"}]],
        "negative_messages": [[{"role": "user", "content": "negative cooking"}]],
    }
    for split in ("train", "validation"):
        (root / f"{split}.jsonl").write_text((json.dumps(row) + "\n") * 16)
    config = {
        "base_path": str(base),
        "train_data": str(root / "train.jsonl"),
        "validation_data": str(root / "validation.jsonl"),
        "max_length": 64,
        "learning_rate": 1e-3,
        "max_steps": 2,
        "output_dir": str(root / "checkpoints"),
    }
    args = swift_arguments(config)
    options = dict(zip(args[::2], args[1::2], strict=True))
    options = {k: v for k, v in options.items() if not k.startswith(("--quant_", "--bnb_"))}
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
    os.environ["INFONCE_USE_BATCH"] = "false"
    os.environ["INFONCE_HARD_NEGATIVES"] = "1"
    os.environ["INFONCE_TEMPERATURE"] = "0.05"
    from swift.llm import sft_main
    from swift.plugin import extra_callbacks

    print("Starting SWIFT CPU training...", flush=True)

    extra_callbacks.append(deadline_callback(time.time() + 300, root / "progress.json"))
    result = sft_main([item for pair in options.items() for item in pair])
    assert result["global_step"] == 2
    assert complete_checkpoints(config["output_dir"])
    assert json.loads((root / "training_integrity.json").read_text())["adapter_updated"]
    faulthandler.cancel_dump_traceback_later()
    from peft import PeftModel

    selected = complete_checkpoints(config["output_dir"])[-1][1]
    reloaded = PeftModel.from_pretrained(Qwen3ForCausalLM.from_pretrained(base), selected)
    assert (
        sum(p.detach().abs().sum().item() for n, p in reloaded.named_parameters() if "lora_B" in n)
        > 0
    )
    import importlib.metadata

    from packaging.version import Version

    resume_verified = False
    if Version(importlib.metadata.version("torch").split("+")[0]) >= Version("2.6"):
        options["--resume_from_checkpoint"] = str(selected)
        options["--max_steps"] = "3"
        resumed = sft_main([item for pair in options.items() for item in pair])
        assert resumed["global_step"] == 3
        resume_verified = True
    else:
        print("Resume smoke skipped: Transformers requires PyTorch >=2.6 for optimizer state.")
    print(
        json.dumps(
            {
                "passed": True,
                "optimizer_steps": result["global_step"],
                "artifacts": str(root),
                "device": "cpu",
                "quantization_tested": False,
                "resume_verified": resume_verified,
            }
        )
    )


if __name__ == "__main__":
    main()
