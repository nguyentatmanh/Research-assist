"""Qwen3 yes/no reranking with the official prompt and a pinned optional LoRA adapter."""

import json
from pathlib import Path

SYSTEM = (
    "Judge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)
PREFIX = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def pair_token_ids(tokenizer, query, document, instruction, max_length, *, truncate=False):
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    body = tokenizer.encode(
        f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}",
        add_special_tokens=False,
    )
    room = max_length - len(prefix) - len(suffix)
    if room < 1:
        raise ValueError("MAX_LENGTH is smaller than the reranker prompt overhead.")
    if len(body) > room and not truncate:
        raise ValueError("Query/document exceeds MAX_LENGTH; gold truncation is forbidden.")
    return prefix + body[:room] + suffix


def read_reranker_manifest(path):
    root = Path(path)
    value = json.loads((root / "research_model.json").read_text(encoding="utf-8"))
    if (
        value.get("format") != "swift_qwen3_reranker_adapter_v1"
        or value.get("task_type") != "generative_reranker"
        or value.get("model_family") != "qwen3_reranker"
        or value.get("base_model") != "Qwen/Qwen3-Reranker-4B"
        or value.get("fine_tuned") is not True
        or value.get("optimizer_steps", 0) < 1
        or len(value.get("base_revision", "")) != 40
        or value.get("positive_token") != "yes"
        or value.get("negative_token") != "no"
        or not (root / "adapter_config.json").is_file()
        or not (root / "adapter_model.safetensors").is_file()
    ):
        raise ValueError("Invalid trained Qwen3 reranker artifact; base fallback is forbidden.")
    return value


class QwenRerankerScorer:
    def __init__(
        self,
        base_path,
        adapter_path=None,
        *,
        instruction=DEFAULT_INSTRUCTION,
        max_length=512,
        device="cuda:0",
        quantized=True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.instruction, self.max_length = instruction, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(base_path), padding_side="left", trust_remote_code=False, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")
        if (
            self.yes == self.no
            or self.yes is None
            or self.no is None
            or self.tokenizer.encode("yes", add_special_tokens=False) != [self.yes]
            or self.tokenizer.encode("no", add_special_tokens=False) != [self.no]
        ):
            raise ValueError("The tokenizer must represent yes and no as distinct single tokens.")
        options = {
            "trust_remote_code": False,
            "local_files_only": True,
            "attn_implementation": "sdpa",
        }
        if quantized:
            if not device.startswith("cuda"):
                raise ValueError("This NF4 recipe requires a CUDA GPU.")
            options.update(
                {
                    "torch_dtype": torch.float16,
                    "device_map": {"": device},
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    ),
                }
            )
        else:
            options["torch_dtype"] = torch.float32 if device == "cpu" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(str(base_path), **options)
        if not quantized:
            model.to(device)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        self.model = model.eval()

    def score(self, pairs, batch_size=1, *, truncate=False):
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        scores = []
        with torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                ids = [
                    pair_token_ids(
                        self.tokenizer, q, d, self.instruction, self.max_length, truncate=truncate
                    )
                    for q, d in pairs[start : start + batch_size]
                ]
                batch = self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
                batch = batch.to(self.model.device)
                # Left padding leaves the final position valid for every pair.
                logits = (
                    self.model(**batch, logits_to_keep=1, use_cache=False).logits[:, -1].float()
                )
                margin = logits[:, self.yes] - logits[:, self.no]
                if not torch.isfinite(margin).all():
                    raise RuntimeError("Non-finite reranker score.")
                scores.extend(margin.cpu().tolist())
        return scores


def load_reranker_adapter(path, device="cuda:0"):
    from huggingface_hub import snapshot_download

    manifest = read_reranker_manifest(path)
    base = snapshot_download(manifest["base_model"], revision=manifest["base_revision"])
    return QwenRerankerScorer(
        base,
        path,
        instruction=manifest["instruction"],
        max_length=manifest["max_length"],
        device=device,
    )
