"""Create small authored fixtures for CLI checks and pipeline smoke training."""

import json

from training_lab.engine.contracts import LAB


def examples(group, subject, metric, value, *, language):
    vietnamese = language == "vi"
    question = (
        f"Thí nghiệm {subject} báo cáo {metric} là bao nhiêu?"
        if vietnamese
        else f"What {metric} did the {subject} experiment report?"
    )
    quote = (
        f"Thí nghiệm {subject} báo cáo {metric} là {value}."
        if vietnamese
        else f"The {subject} experiment reported {metric} of {value}."
    )
    chunk = "chunk_" + group
    unit = "unit_" + group
    doc = "doc_" + group
    locator = {"chunk_id": chunk, "unit_id": unit, "start": 0, "end": len(quote)}
    source = {
        "chunk_id": chunk,
        "page": 2,
        "section": "Results",
        "filename": group + ".pdf",
        "title": subject,
    }
    evidence = {
        **locator,
        "id": "e1",
        "document_id": doc,
        "quote": quote,
        "relevance": "Direct result.",
        "document_title": subject,
        "filename": source["filename"],
        "page": 2,
        "section": "Results",
    }
    req = {"id": "rq_1", "text": question}
    claim = {"text": quote, "evidence_ids": ["e1"], "requirement_ids": ["rq_1"]}
    verified = {**claim, "id": "c1"}
    result = {
        "question_plan": (
            {
                "question": question,
                "requirements": [],
                "sources": [{"id": doc, "title": subject, "filename": source["filename"]}],
            },
            {
                "aspects": [
                    {"text": question, "query": f"{subject} {metric}", "document_ids": [doc]}
                ]
            },
        ),
        "read_evidence": (
            {
                "question": question,
                "requirements": [question],
                "reading_requests": [],
                "source_units": [{**locator, "text": quote}],
                "sources": [source],
            },
            {
                "evidence": [
                    {
                        "chunk_id": chunk,
                        "unit_id": unit,
                        "requirement_ids": ["rq_1"],
                        "relevance": "Direct result.",
                    }
                ]
            },
        ),
        "synthesize_claims": (
            {
                "question": question,
                "language": language,
                "requirements": [req],
                "evidence": [evidence],
                "revision_requests": [],
            },
            {"claims": [claim]},
        ),
        "check_claims": (
            {
                "question": question,
                "candidate_gaps": [],
                "claims": [
                    verified,
                    {
                        **claim,
                        "id": "c2",
                        "text": quote + " The article reports an independent replication.",
                    },
                ],
                "requirements": [req],
                "evidence": [evidence],
            },
            {
                "checks": [
                    {
                        "claim_id": "c1",
                        "verdict": "supported",
                        "evidence_ids": ["e1"],
                        "explanation": "The quote reports this result.",
                    },
                    {
                        "claim_id": "c2",
                        "verdict": "insufficient",
                        "evidence_ids": ["e1"],
                        "explanation": "The quote does not mention replication.",
                    },
                ],
                "coverage": [
                    {
                        "requirement_id": "rq_1",
                        "status": "covered",
                        "claim_ids": ["c1"],
                        "explanation": "Reported result.",
                    }
                ],
            },
        ),
        "write_report": (
            {
                "question": question,
                "language": language,
                "style": "overview",
                "requirements": [question],
                "verified_claims": [verified],
            },
            {
                "title": question,
                "sections": [
                    {
                        "heading": "Kết quả" if vietnamese else "Results",
                        "blocks": [{"claim_id": "c1"}],
                    }
                ],
            },
        ),
    }
    for task, (payload, expected) in result.items():
        yield {"task": task, "group": group, "payload": payload, "expected": expected}

    for task, role in (
        ("question_plan", "retrieval"),
        ("read_evidence", "reader"),
        ("synthesize_claims", "summarizer"),
        ("check_claims", "fact_checker"),
        ("write_report", "writer"),
    ):
        payload, expected = result[task]
        context = {**payload, "assignment": {"role": role, "requirement_ids": ["rq_1"]}}
        if role == "retrieval":
            context["observed_chunks"] = [{"id": chunk, "snippet": quote}]
            expected = {"chunk_ids": [chunk]}
        yield {
            "task": role + "_step",
            "group": group,
            "payload": {
                "task_id": group + "_" + role,
                "context": context,
                "tools": [],
                "memory": [],
                "last_observation": {},
                "steps_remaining": 1,
            },
            "expected": {
                "action": "finish",
                "result": expected,
                "reason": "Finished the assigned task.",
            },
        }


def main():
    groups = {
        "train": [
            ("train-sensors", "Sensor Atlas", "sampling interval", "20 ms", "en"),
            ("train-search", "Tra cứu Sông Hồng", "Recall@10", "83%", "vi"),
        ],
        "validation": [("validation-water", "Water Twin", "F1", "0.76", "en")],
        "test": [("test-traffic", "Điều phối Đường 3", "độ trễ trung vị", "110 ms", "vi")],
    }
    for split, topics in groups.items():
        records = [
            row
            for group, subject, metric, value, language in topics
            for row in examples(group, subject, metric, value, language=language)
        ]
        (LAB / "data" / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
