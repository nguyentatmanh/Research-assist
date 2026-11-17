"""Train a PEFT adapter on assistant tokens only, using a held-out validation split."""

import argparse
import json
from pathlib import Path
from typing import Any

from training_lab.engine.contracts import LAB, ROLES, Example, for_role, load_splits, split_hashes


def encode_examples(
    rows: list[Example], tokenizer: Any, max_length: int
) -> list[dict[str, list[int]]]:
    encoded = []
    for row in rows:
        prefix = tokenizer.apply_chat_template(
            row.messages(completion=False), tokenize=True, add_generation_prompt=True
        )
        complete = tokenizer.apply_chat_template(row.messages(), tokenize=True)
        if complete[: len(prefix)] != prefix or len(complete) <= len(prefix):
            raise ValueError("Tokenizer chat template must preserve the assistant prompt prefix")
        if len(complete) > max_length:
            continue
        encoded.append(
            {"input_ids": complete, "labels": [-100] * len(prefix) + complete[len(prefix) :]}
        )
    return encoded


def main(default_role: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    if default_role is None:
        parser.add_argument("--role", choices=ROLES, required=True)
    else:
        parser.set_defaults(role=default_role)
    parser.add_argument("--dataset", type=Path, default=LAB / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--allow-demo-data", action="store_true")
    args = parser.parse_args()

    if args.dataset.resolve() == (LAB / "data").resolve() and not args.allow_demo_data:
        parser.error(
            "The bundled examples are a smoke fixture. Use --allow-demo-data to test training."
        )
    if min(args.max_length, args.epochs, args.batch_size, args.accumulation) <= 0:
        parser.error("Training sizes and epochs must be positive")
    output = args.output or LAB / "final_weights" / args.role
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Choose an empty output directory")
    splits = load_splits(args.dataset)
    rows = {name: for_role(splits[name], args.role) for name in ("train", "validation")}
    if not all(rows.values()):
        parser.error("This role needs examples in both train and validation")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if args.qlora and torch.cuda.device_count() != 1:
        parser.error("QLoRA expects exactly one visible GPU. Select it with CUDA_VISIBLE_DEVICES.")
    set_seed(args.seed)
    options = {"revision": args.revision, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(args.base, **options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = {
        name: encode_examples(items, tokenizer, args.max_length) for name, items in rows.items()
    }
    for name, items in encoded.items():
        print(f"{name}: {len(items)}/{len(rows[name])} samples fit without truncation")
        if not items:
            parser.error(f"No {name} samples fit --max-length")

    model_options: dict[str, Any] = dict(options)
    if args.qlora:
        model_options.update(
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            ),
        )
    model = AutoModelForCausalLM.from_pretrained(args.base, **model_options)
    revision = getattr(model.config, "_commit_hash", None) or args.revision
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora, revision=revision)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output / "checkpoints"),
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.accumulation,
            gradient_checkpointing=args.qlora,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            prediction_loss_only=True,
            dataloader_pin_memory=torch.cuda.is_available(),
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=Dataset.from_list(encoded["train"]),
        eval_dataset=Dataset.from_list(encoded["validation"]),
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
    )
    trainer.train()
    adapter = output / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    manifest = {
        "role": args.role,
        "base_model": args.base,
        "revision": revision,
        "seed": args.seed,
        "qlora": args.qlora,
        "dataset_sha256": split_hashes(args.dataset),
        "samples": {name: len(items) for name, items in encoded.items()},
        "validation_metrics": trainer.evaluate(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
