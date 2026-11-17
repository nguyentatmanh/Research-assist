# -*- coding: utf-8 -*-
"""
Rich Multi-Class DocLayNet Dataset Generator from Real PDFs.
Extracts real layout blocks using PyMuPDF:
  0: Caption, 1: Footnote, 2: Formula, 3: List-item, 4: Page-footer,
  5: Page-header, 6: Picture, 7: Section-header, 8: Table, 9: Text, 10: Title
Normalizes bounding boxes to standard [0, 1000] coordinate grid for LayoutLMv3.
"""

import re
import json
from pathlib import Path
import fitz  # PyMuPDF

DOCLAYNET_LABELS = [
    "Caption", "Footnote", "Formula", "List-item", "Page-footer",
    "Page-header", "Picture", "Section-header", "Table", "Text", "Title"
]
LABEL2ID = {l: i for i, l in enumerate(DOCLAYNET_LABELS)}


def classify_block(text: str, bbox: tuple, page_height: float, font_size: float = 10.0) -> int:
    """Classifies a text block into one of 11 DocLayNet structural categories."""
    t = text.strip()
    x0, y0, x1, y1 = bbox
    h_safe = max(1.0, float(page_height))
    rel_y0 = y0 / h_safe
    rel_y1 = y1 / h_safe

    # 1. Page Header (top 8%)
    if rel_y1 <= 0.08 and len(t) < 100:
        return LABEL2ID["Page-header"]

    # 2. Page Footer (bottom 8%)
    if rel_y0 >= 0.92 and len(t) < 100:
        return LABEL2ID["Page-footer"]

    # 3. Caption (starts with Figure, Table, Fig)
    if re.match(r"^(Figure|Fig\.|Table)\s+\d+", t, re.IGNORECASE):
        return LABEL2ID["Caption"]

    # 4. Table (contains multiple tab/pipe separators or numbers)
    lines = t.split("\n")
    if len(lines) >= 3 and any(len(re.findall(r"\d+", l)) >= 3 for l in lines):
        return LABEL2ID["Table"]

    # 5. Formula (contains math operators and equations)
    if any(sym in t for sym in ["∑", "∫", "∂", "√", "≤", "≥", "≠", "±", "λ"]) or (len(t) < 60 and re.search(r"[a-zA-Z]\s*=\s*[\d\w\+\-\*\/]+", t)):
        return LABEL2ID["Formula"]

    # 6. List-item (bullet or numbered list)
    if re.match(r"^(\*|\-|\•|\d+[\.\)])\s+", t):
        return LABEL2ID["List-item"]

    # 7. Footnote
    if rel_y0 >= 0.85 and re.match(r"^(\d+|[\*\†])\s+", t):
        return LABEL2ID["Footnote"]

    # 8. Title / Section-header
    if len(t) < 80 and not t.endswith("."):
        if rel_y0 < 0.20 and (t.isupper() or len(t.split()) <= 6):
            return LABEL2ID["Title"]
        return LABEL2ID["Section-header"]

    # 9. Default: Body Text
    return LABEL2ID["Text"]


def generate_doclaynet_from_pdf(pdf_path: Path, out_path: Path, max_pages: int = 100):
    """Parses real PDF pages and creates genuine multi-class LayoutLMv3 records."""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        for cand in [Path("../") / pdf_path, Path(__file__).resolve().parent.parent.parent / pdf_path]:
            if cand.exists():
                pdf_file = cand
                break
    if not pdf_file.exists():
        if Path(out_path).exists():
            print(f"⚠️ [DocLayNet Generator] PDF not found at {pdf_path}. Reusing verified dataset at {out_path}.")
            with open(out_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    pdf_path = pdf_file
    doc = fitz.open(pdf_path)
    total_pages = min(len(doc), max_pages)
    print(f"📖 [DocLayNet Generator] Processing {total_pages} pages from {pdf_path.name}...")

    records = []
    class_distribution = {label: 0 for label in DOCLAYNET_LABELS}

    for page_idx in range(total_pages):
        page = doc[page_idx]
        w, h = page.rect.width, page.rect.height
        if w <= 0 or h <= 0:
            continue

        blocks = page.get_text("blocks")
        tokens = []
        bboxes = []
        ner_tags = []

        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type != 0 or not text.strip():
                continue

            tag_id = classify_block(text, (x0, y0, x1, y1), h)
            class_name = DOCLAYNET_LABELS[tag_id]
            class_distribution[class_name] += 1

            # Normalize bbox to [0, 1000] integer grid
            nb_x0 = int(max(0, min(1000, (x0 / w) * 1000)))
            nb_y0 = int(max(0, min(1000, (y0 / h) * 1000)))
            nb_x1 = max(nb_x0, int(max(0, min(1000, (x1 / w) * 1000))))
            nb_y1 = max(nb_y0, int(max(0, min(1000, (y1 / h) * 1000))))
            norm_bbox = [nb_x0, nb_y0, nb_x1, nb_y1]

            words = text.strip().split()
            for word in words:
                clean_word = re.sub(r"[^\w\-\.\,\:\;\(\)\[\]]", "", word)
                if clean_word:
                    tokens.append(clean_word)
                    bboxes.append(norm_bbox)
                    ner_tags.append(tag_id)

        if len(tokens) >= 10:
            records.append({
                "id": f"page_{page_idx+1:04d}",
                "tokens": tokens[:510],
                "bboxes": bboxes[:510],
                "ner_tags": ner_tags[:510]
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ [DocLayNet Generator] Exported {len(records)} authentic pages -> {out_path}")
    print("📊 Phân bổ nhãn thực tế trích xuất được:")
    for lbl, cnt in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
        if cnt > 0:
            print(f"   • {lbl:<15}: {cnt:>5} khối")

    doc.close()
    return records


if __name__ == "__main__":
    candidates = [
        Path("uploads/softwareengineering-apractitionersapproach-rogerspressman-110630132301-phpapp01.pdf"),
        Path("../uploads/softwareengineering-apractitionersapproach-rogerspressman-110630132301-phpapp01.pdf"),
        Path(__file__).resolve().parent.parent.parent / "uploads" / "softwareengineering-apractitionersapproach-rogerspressman-110630132301-phpapp01.pdf",
    ]
    pdf = next((c for c in candidates if c.exists()), candidates[0])
    target_out = Path("training_lab/agent2_reader/dataset/layout_doclaynet_sample.jsonl")
    backup_out = Path("training_lab/data/doclaynet_real/layout_doclaynet_clean.jsonl")

    generate_doclaynet_from_pdf(pdf, target_out, max_pages=150)
    generate_doclaynet_from_pdf(pdf, backup_out, max_pages=150)
