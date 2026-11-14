"""Source-only checklist inspired by the upstream statistical auditor.

Mentions are not rigor scores. A missing marker in retrieved snippets does not
imply its absence in the whole paper. Each observed marker has a source locator.
"""

import re

from research_assist.domain.models import Evidence

PATTERNS = {
    "seeds": r"\b(?:seeds?|random states?|hạt giống)\b",
    "uncertainty": r"(?:±|confidence interval|standard deviation|độ lệch chuẩn|khoảng tin cậy)",
    "test": r"(?:p\s*[<=>]\s*0\.\d+|wilcoxon|bootstrap|paired t.test)",
    "sample_size": r"(?:\bn\s*=\s*\d+|\b\d+\s*(?:samples|examples|questions|mẫu))",
    "hardware": r"\b(?:GPU|CPU|T4|A100|H100|V100|VRAM|compute budget)\b",
    "limitations": r"\b(?:limitations?|hạn chế|future work)\b",
}


def methodology_checklist(evidence: list[Evidence]) -> dict[str, object]:
    return {
        "scope": "retrieved_evidence_only",
        "warning": "Chỉ phát hiện từ khóa trong bằng chứng đã đọc; không chấm chất lượng bài báo.",
        "items": [
            {
                "criterion": name,
                "matches": [
                    {
                        "evidence_id": e.id,
                        "document_id": e.document_id,
                        "page": e.page,
                        "quote": e.quote,
                        "start": e.start,
                        "end": e.end,
                    }
                    for e in evidence
                    if re.search(pattern, e.quote, re.I)
                ],
            }
            for name, pattern in PATTERNS.items()
        ],
    }
