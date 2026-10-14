"""Convert official raw QASPER JSON into local text documents and benchmark JSONL.

The input should be the official raw mapping: {paper_id: {full_text: [...], qas: [...]}}.
No automatic downloads. Keep original dataset license and train/dev/test split boundaries.
"""

import argparse
import json
import re
from pathlib import Path


def convert(source: Path, output: Path, max_papers: int) -> int:
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            "Expected the official QASPER raw JSON mapping, not the HF columnar format"
        )
    documents = output / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    cases = []
    for paper_id, paper in list(data.items())[:max_papers]:
        slug = re.sub(r"[^A-Za-z0-9_-]", "_", str(paper_id))
        filename = f"qasper_{slug}.txt"
        sections = []
        for section in paper.get("full_text", []):
            sections.append(section.get("section_name", ""))
            sections.extend(section.get("paragraphs", []))
        text = "\n\n".join(
            [paper.get("title", str(paper_id)), paper.get("abstract", ""), *sections]
        )
        (documents / filename).write_text(text, encoding="utf-8")
        for qa in paper.get("qas", []):
            annotations = [entry.get("answer", {}) for entry in qa.get("answers", [])]
            answerable = [a for a in annotations if not a.get("unanswerable", False)]
            gold_evidence = list(
                dict.fromkeys(e for a in answerable for e in a.get("evidence", []))
            )
            cases.append(
                {
                    "id": qa.get("question_id", f"{slug}_{len(cases)}"),
                    "question": qa["question"],
                    "scope_files": [filename],
                    "relevant_files": [filename] if answerable else [],
                    "unanswerable": not bool(answerable),
                    "gold_evidence": gold_evidence,
                    "gold_answers": answerable,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "questions.jsonl").write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n", encoding="utf-8"
    )
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation/qasper-dev"))
    parser.add_argument("--max-papers", type=int, default=20)
    args = parser.parse_args()
    print(f"Converted {convert(args.source, args.output, args.max_papers)} questions")


if __name__ == "__main__":
    main()
