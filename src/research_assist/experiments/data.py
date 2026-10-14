"""QASPER paragraphs and gold evidence, preserving official paper-level splits."""

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def records(value: Any) -> list[dict[str, Any]]:
    """HF's columnar Sequence and official raw list have the same canonical representation."""
    if isinstance(value, list):
        return value
    if not value:
        return []
    lengths = {len(v) for v in value.values()}
    if len(lengths) != 1:
        raise ValueError("Malformed columnar sequence")
    return [{k: v[i] for k, v in value.items()} for i in range(next(iter(lengths)))]


def prepare_qasper(source: Path, output: Path) -> dict[str, Any]:
    corpus: list[dict[str, Any]] = []
    queries: dict[str, list[dict[str, Any]]] = {}
    pairs: list[dict[str, Any]] = []
    paper_splits: dict[str, str] = {}
    stats: Counter[str] = Counter()
    hashes: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        path = source / f"{split}.json"
        hashes[split] = file_hash(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        papers = raw.items() if isinstance(raw, dict) else ((r["id"], r) for r in raw)
        queries[split] = []
        for paper_id, paper in papers:
            paper_id = str(paper_id)
            if paper_id in paper_splits:
                raise ValueError(f"Paper leakage or duplicate: {paper_id}")
            paper_splits[paper_id] = split
            stats[f"{split}_papers"] += 1
            lookup: dict[str, list[str]] = {}
            paragraphs = [("Abstract", paper.get("abstract", ""))]
            for section in records(paper["full_text"]):
                paragraphs.extend(
                    (section.get("section_name", ""), p) for p in section["paragraphs"]
                )
            for index, (section_name, text) in enumerate(paragraphs):
                if not text.strip():
                    continue
                cid = paper_id + ":" + str(index)
                row = {
                    "id": cid,
                    "paper_id": paper_id,
                    "text": text.strip(),
                    "section": section_name,
                    "split": split,
                }
                corpus.append(row)
                lookup.setdefault(normalize(text), []).append(cid)
            for qa in records(paper["qas"]):
                answers = [a["answer"] for a in records(qa["answers"])]
                answerable = [a for a in answers if not a.get("unanswerable", False)]
                gold = list(dict.fromkeys(e for a in answerable for e in a.get("evidence", [])))
                textual = [e for e in gold if not e.startswith("FLOAT SELECTED")]
                ids = sorted({cid for e in textual for cid in lookup.get(normalize(e), [])})
                unmatched = [e for e in textual if normalize(e) not in lookup]
                # Partially mapped gold would inflate recall: exclude such questions, report why.
                reason = (
                    "unanswerable"
                    if not answerable
                    else "non_text_evidence"
                    if len(textual) != len(gold)
                    else "unmatched_gold"
                    if unmatched
                    else "no_gold"
                    if not ids
                    else ""
                )
                question = {
                    "id": str(qa["question_id"]),
                    "question": qa["question"],
                    "paper_id": paper_id,
                    "relevant_ids": ids,
                    "excluded_reason": reason,
                    "split": split,
                }
                queries[split].append(question)
                stats[f"{split}_{reason or 'eligible'}"] += 1
                if split == "train" and not reason:
                    pairs.extend(
                        {
                            "query_id": question["id"],
                            "question": question["question"],
                            "positive_id": cid,
                            "paper_id": paper_id,
                            "source": "qasper_gold",
                            "reviewed": True,
                        }
                        for cid in ids
                    )
    # Exact duplicate paragraphs across papers can contaminate evaluation too.
    train_texts = {normalize(c["text"]) for c in corpus if c["split"] == "train"}
    heldout_texts = {normalize(c["text"]) for c in corpus if c["split"] != "train"}
    training_text = {c["id"]: normalize(c["text"]) for c in corpus if c["split"] == "train"}
    original_pairs = len(pairs)
    pairs = [p for p in pairs if training_text[p["positive_id"]] not in heldout_texts]
    stats["train_pairs_excluded_duplicate_text"] = original_pairs - len(pairs)
    stats["train_pairs"] = len(pairs)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "corpus.jsonl", corpus)
    write_jsonl(output / "train_pairs.jsonl", pairs)
    for split, rows in queries.items():
        write_jsonl(output / f"{split}_queries.jsonl", rows)
    manifest = {
        "format_version": 2,
        "dataset": "allenai/qasper",
        "language": "en",
        "source_hashes": hashes,
        "paper_split": paper_splits,
        "statistics": dict(stats),
        "cross_split_duplicate_texts": len(train_texts & heldout_texts),
        "corpus_sha256": file_hash(output / "corpus.jsonl"),
        "pairs_sha256": file_hash(output / "train_pairs.jsonl"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split(data: Path, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus = [r for r in read_jsonl(data / "corpus.jsonl") if r["split"] == split]
    queries = [q for q in read_jsonl(data / f"{split}_queries.jsonl") if not q["excluded_reason"]]
    ids = {r["id"] for r in corpus}
    if len(ids) != len(corpus) or not queries or not corpus:
        raise ValueError("Missing eligible queries/corpus or duplicate corpus IDs")
    if any(not q["relevant_ids"] or not set(q["relevant_ids"]) <= ids for q in queries):
        raise ValueError("Gold evidence outside its split")
    return corpus, queries


def validated_pairs(data: Path, synthetic: Path | None = None) -> list[dict[str, Any]]:
    corpus = {r["id"]: r for r in read_jsonl(data / "corpus.jsonl")}
    heldout_texts = {normalize(c["text"]) for c in corpus.values() if c["split"] != "train"}
    pairs = read_jsonl(data / "train_pairs.jsonl")
    if synthetic:
        pairs += read_jsonl(synthetic)
    result, seen = [], set()
    for pair in pairs:
        if pair.get("reviewed") is not True:
            raise ValueError("Synthetic data must be reviewed by a human before training")
        positive = corpus.get(pair["positive_id"])
        if not positive or positive["split"] != "train" or positive["paper_id"] != pair["paper_id"]:
            raise ValueError("Training evidence leaks held-out papers or has invalid provenance")
        if normalize(positive["text"]) in heldout_texts:
            raise ValueError("Training evidence duplicates held-out text")
        key = (pair["question"], pair["positive_id"])
        if key not in seen:
            seen.add(key)
            result.append({**pair, "positive": positive["text"]})
    if not result:
        raise ValueError("No valid training pairs")
    return result
