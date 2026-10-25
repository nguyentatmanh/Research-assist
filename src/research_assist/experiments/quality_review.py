"""Bind independent human scores to the exact answer that was reviewed."""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import Field

from research_assist.domain.models import Contract


class Review(Contract):
    case_id: str
    answer_sha256: str
    reviewer: str = Field(min_length=1)
    faithfulness: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    correct_abstention: bool | None = None
    notes: str = ""


def review_template(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": row["id"],
            "answer_sha256": row.get("answer_sha256"),
            "reviewer": "",
            "faithfulness": None,
            "completeness": None,
            "correct_abstention": None,
            "notes": "",
            "question": row["question"],
            "reference_answer": row.get("reference_answer", ""),
        }
        for row in benchmark["cases"]
        if row.get("answer_sha256")
    ]


def apply_reviews(benchmark: dict[str, Any], reviews: list[Review]) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(json.dumps(benchmark))
    rows = {row["id"]: row for row in result["cases"]}
    if len({r.case_id for r in reviews}) != len(reviews):
        raise ValueError("Duplicate review case ID")
    for review in reviews:
        if review.case_id not in rows:
            raise ValueError("Review references an unknown case")
        row = rows[review.case_id]
        if row.get("answer_sha256") != review.answer_sha256:
            raise ValueError("Answer changed since review; review the current answer again")
        row.update(
            human_faithfulness=review.faithfulness,
            human_answer_completeness=review.completeness,
            human_correct_abstention=review.correct_abstention,
            human_review=review.model_dump(),
        )
    scored = [row for row in rows.values() if row.get("human_review")]
    result["summary"].update(
        human_reviewed_cases=len(scored),
        human_review_total_cases=len(rows),
        human_faithfulness=mean(row["human_faithfulness"] for row in scored) if scored else None,
        human_answer_completeness=mean(row["human_answer_completeness"] for row in scored)
        if scored
        else None,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    if args.reviews:
        reviews = []
        for line in args.reviews.read_text(encoding="utf-8").splitlines():
            if line.strip():
                data = json.loads(line)
                data.pop("question", None)
                data.pop("reference_answer", None)
                if (
                    not data.get("reviewer")
                    and data.get("faithfulness") is None
                    and data.get("completeness") is None
                ):
                    continue
                reviews.append(Review.model_validate(data))
        content = json.dumps(apply_reviews(benchmark, reviews), ensure_ascii=False, indent=2)
    else:
        content = "\n".join(
            json.dumps(row, ensure_ascii=False) for row in review_template(benchmark)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
