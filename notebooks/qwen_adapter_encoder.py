"""Inference for the SWIFT Qwen3 adapter recipe; identical query/EOS policy to training."""

import hashlib
import json
from pathlib import Path


def adapter_identity(path, max_seq_length):
    root = Path(path)
    digest = hashlib.sha256(f"swift-qwen3-eos-v1:{max_seq_length}".encode())
    for name in ("research_model.json", "adapter_config.json", "adapter_model.safetensors"):
        digest.update(name.encode())
        with (root / name).open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return "research-qwen3-" + digest.hexdigest()[:32]


def read_adapter_manifest(path):
    root = Path(path)
    manifest = json.loads((root / "research_model.json").read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "swift_qwen3_adapter_v1"
        or manifest.get("model_family") != "qwen3"
        or manifest.get("fine_tuned") is not True
        or manifest.get("optimizer_steps", 0) < 1
        or manifest.get("base_model")
        not in {"Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-0.6B"}
        or len(manifest.get("base_revision", "")) != 40
        or not manifest.get("query_instruction")
        or not (root / "adapter_model.safetensors").is_file()
    ):
        raise ValueError("Invalid trained Qwen3 adapter manifest; base fallback is forbidden.")
    return manifest


def extract_adapter_archive(archive_path, destination):
    import zipfile

    root = Path(destination).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Choose an empty adapter extraction directory.")
    with zipfile.ZipFile(archive_path) as archive:
        size = 0
        for item in archive.infolist():
            target = (root / item.filename.replace("\\", "/")).resolve()
            if not target.is_relative_to(root) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe adapter archive entry.")
            size += item.file_size
            if size > 2 * 1024**3:
                raise ValueError("Adapter archive exceeds 2 GiB.")
        archive.extractall(root)
    read_adapter_manifest(root)
    return root


class QwenAdapterEncoder:
    """Load a pinned base plus optional adapter. Use base-only solely for baseline evaluation."""

    def __init__(self, base_path, instruction, max_seq_length=512, adapter_path=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.max_seq_length, self.instruction = max_seq_length, instruction
        self.tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=False)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.suffix = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if self.suffix is None or self.suffix == self.tokenizer.unk_token_id:
            raise ValueError("Qwen tokenizer must contain the SWIFT endoftext suffix.")
        model = AutoModelForCausalLM.from_pretrained(
            str(base_path),
            trust_remote_code=False,
            local_files_only=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        self.model = model.eval()

    def get_sentence_embedding_dimension(self):
        return int(self.model.config.hidden_size)

    def encode_query(self, texts, **kwargs):
        return self._encode(
            [f"Instruct: {self.instruction}\nQuery:{text}" for text in texts], **kwargs
        )

    def encode_document(self, texts, **kwargs):
        return self._encode(texts, **kwargs)

    def _encode(self, texts, batch_size=2, **kwargs):
        import numpy as np
        import torch
        import torch.nn.functional as functional

        result = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                ids = self.tokenizer(
                    texts[start : start + batch_size],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_seq_length - 1,
                )["input_ids"]
                batch = self.tokenizer.pad(
                    {"input_ids": [row + [self.suffix] for row in ids]},
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
                base = (
                    self.model.get_base_model()
                    if hasattr(self.model, "peft_config")
                    else self.model
                )
                # Use the transformer with adapters; avoid the vocabulary-sized LM head.
                hidden = base.model(**batch, use_cache=False, return_dict=True).last_hidden_state
                last = batch["attention_mask"].sum(dim=1) - 1
                vectors = hidden[torch.arange(len(last), device=hidden.device), last].float()
                result.append(functional.normalize(vectors, dim=1).cpu().numpy())
        return (
            np.concatenate(result)
            if result
            else np.empty((0, self.get_sentence_embedding_dimension()))
        )


def load_swift_qwen_adapter(path, max_seq_length=512):
    from huggingface_hub import snapshot_download

    manifest = read_adapter_manifest(path)
    base = snapshot_download(manifest["base_model"], revision=manifest["base_revision"])
    encoder = QwenAdapterEncoder(
        base, manifest["query_instruction"], max_seq_length, adapter_path=path
    )
    return encoder, adapter_identity(path, max_seq_length)
