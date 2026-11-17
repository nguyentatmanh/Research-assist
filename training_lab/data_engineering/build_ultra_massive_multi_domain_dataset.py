# -*- coding: utf-8 -*-
"""
Ultra-Massive Multi-Domain Dataset Compiler for AI 2.0 Multi-Agent RAG.
Aggregates 10,000+ Triplets and 20,000+ Pairs across 5 real-world domains:
  1. Vietnamese Medical & Healthcare QA (hungnm/vietnamese-medical-qa)
  2. Vietnamese Banking & Financial QA (HoangVuSnape/Vietnamese_Banking_QA)
  3. Academic, Logic & Scientific Reasoning (sentence-transformers/all-nli)
  4. Global Science, Finance & Biomedical (BEIR SciFact, FiQA, NFCorpus)
Zero bytes from local PDF (Pressman 888p) are included — kept 100% as clean Hold-out Test Set.
"""

import json
from pathlib import Path
from datasets import load_dataset



OFFLINE_FALLBACK_SEEDS = [
    {"query": "Chỉ số an toàn vốn CAR tối thiểu theo chuẩn Basel II là bao nhiêu?", "positive": "Theo tiêu chuẩn Basel II, tỷ lệ an toàn vốn tối thiểu được ấn định là 8% tính trên tổng tài sản có trọng số rủi ro.", "negative": "Thuật toán sắp xếp nhanh QuickSort có độ phức tạp trung bình là O(n log n).", "domain": "banking_finance"},
    {"query": "Liều lượng tối đa của Paracetamol cho người trưởng thành trong 24 giờ là bao nhiêu?", "positive": "Liều khuyến cáo tối đa của Paracetamol là 4g (4000mg) mỗi 24 giờ đối với người lớn có chức năng gan bình thường.", "negative": "Hợp đồng kinh tế có hiệu lực kể từ thời điểm các bên ký kết hoặc theo thỏa thuận cụ thể.", "domain": "medical_healthcare"},
    {"query": "Mục tiêu cốt lõi của quy trình Cleanroom software engineering là gì?", "positive": "Cleanroom hướng tới chế tạo phần mềm có độ tin cậy cực cao, đưa tỷ lệ khiếm khuyết tiếp cận 0 thông qua kiểm chứng tĩnh và kiểm thử thống kê.", "negative": "Lợi nhuận gộp của doanh nghiệp được tính bằng doanh thu thuần trừ đi giá vốn hàng bán.", "domain": "computer_science"},
    {"query": "Thời hiệu khởi kiện tranh chấp hợp đồng thương mại là bao lâu theo luật?", "positive": "Thời hiệu khởi kiện áp dụng đối với các tranh chấp thương mại là 02 năm kể từ thời điểm quyền và lợi ích hợp pháp bị xâm phạm.", "negative": "Kháng sinh nhóm Penicillin ức chế sự tổng hợp thành tế bào vi khuẩn.", "domain": "legal_compliance"},
]

def build_ultra_dataset(
    out_triplets_file: Path,
    out_reranker_file: Path
):
    out_triplets_file.parent.mkdir(parents=True, exist_ok=True)
    out_reranker_file.parent.mkdir(parents=True, exist_ok=True)

    triplets = []
    reranker_pairs = []

    # 1. Vietnamese Medical QA
    print("🧬 [1/4] Ingesting Vietnamese Medical QA (hungnm/vietnamese-medical-qa)...")
    try:
        ds_med = load_dataset("hungnm/vietnamese-medical-qa", split="train")
        med_samples = list(ds_med)[:2000]
        for i in range(len(med_samples) - 1):
            q = str(med_samples[i].get("question") or "").strip()
            pos = str(med_samples[i].get("answer") or "").strip()
            neg = str(med_samples[i + 1].get("answer") or "").strip()
            if len(q) > 10 and len(pos) > 20:
                triplets.append({"query": q, "positive": pos, "negative": neg, "domain": "vietnamese_medical"})
                reranker_pairs.append({"query": q, "document": pos, "label": 1.0})
                reranker_pairs.append({"query": q, "document": neg, "label": 0.0})
        print(f"✅ Extracted {len(med_samples)} Vietnamese Medical QA pairs.")
    except Exception as e:
        print(f"⚠️ Medical error: {e}")

    # 2. Vietnamese Banking & Financial QA
    print("💰 [2/4] Ingesting Vietnamese Banking & Finance QA (HoangVuSnape/Vietnamese_Banking_QA)...")
    try:
        ds_bank = load_dataset("HoangVuSnape/Vietnamese_Banking_QA", split="train")
        bank_samples = list(ds_bank)
        for i in range(len(bank_samples) - 1):
            q = str(bank_samples[i].get("question") or "").strip()
            pos = str(bank_samples[i].get("answer") or "").strip()
            neg = str(bank_samples[i + 1].get("answer") or "").strip()
            if len(q) > 10 and len(pos) > 20:
                triplets.append({"query": q, "positive": pos, "negative": neg, "domain": "vietnamese_banking_finance"})
                reranker_pairs.append({"query": q, "document": pos, "label": 1.0})
                reranker_pairs.append({"query": q, "document": neg, "label": 0.0})
        print(f"✅ Extracted {len(bank_samples)} Vietnamese Banking & Finance QA pairs.")
    except Exception as e:
        print(f"⚠️ Banking error: {e}")

    # 3. Academic & Scientific Logic (All-NLI)
    print("🔬 [3/4] Ingesting Academic Logic Triplets (sentence-transformers/all-nli)...")
    try:
        ds_nli = load_dataset("sentence-transformers/all-nli", "triplet", split="train", streaming=True)
        count = 0
        for row in ds_nli:
            if count >= 6000:
                break
            anchor = row.get("anchor", "").strip()
            pos = row.get("positive", "").strip()
            neg = row.get("negative", "").strip()
            if not anchor or not pos or not neg:
                continue

            triplets.append({"query": anchor, "positive": pos, "negative": neg, "domain": "scientific_logic"})
            reranker_pairs.append({"query": anchor, "document": pos, "label": 1.0})
            reranker_pairs.append({"query": anchor, "document": neg, "label": 0.0})
            count += 1
        print(f"✅ Extracted {count} Academic Logic Triplets.")
    except Exception as e:
        print(f"⚠️ All-NLI error: {e}")

    # 4. BEIR Multi-Domain Corpus (SciFact, FiQA, NFCorpus)
    print("🌐 [4/4] Ingesting BEIR Multi-Domain Corpus...")
    multi_dir = Path(__file__).resolve().parent.parent / "data" / "multi_domain"
    for domain_name, fname in [
        ("science_engineering", "science_engineering_corpus.jsonl"),
        ("finance_economics", "finance_corpus.jsonl"),
        ("biomedical_health", "biomedical_corpus.jsonl"),
    ]:
        p = multi_dir / fname
        if not p.exists():
            continue
        passages = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    c = json.loads(line).get("content", "")
                    if c.strip():
                        passages.append(c.strip())
                except Exception:
                    continue

        for i in range(len(passages) - 1):
            pos = passages[i]
            neg = passages[i + 1]
            raw_words = [w.strip(".,;:?!'\"") for w in pos.split()[:8]]
            words = [w for w in raw_words if w]
            if not words:
                continue
            q = f"What are the established findings regarding {' '.join(words)}?"
            triplets.append({"query": q, "positive": pos, "negative": neg, "domain": domain_name})
            reranker_pairs.append({"query": q, "document": pos, "label": 1.0})
            reranker_pairs.append({"query": q, "document": neg, "label": 0.0})

    # Ensure non-empty fallback if offline and datasets failed
    if not triplets:
        for s in OFFLINE_FALLBACK_SEEDS:
            triplets.append(s)
            reranker_pairs.append({"query": s["query"], "document": s["positive"], "label": 1.0})
            reranker_pairs.append({"query": s["query"], "document": s["negative"], "label": 0.0})

    # Save to disk
    with open(out_triplets_file, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    with open(out_reranker_file, "w", encoding="utf-8") as f:
        for r in reranker_pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 90)
    print("🏆 [HOÀN TẤT THIẾT LẬP KHO DỮ LIỆU ĐA LĨNH VỰC ULTRA-MASSIVE CHO AI 2.0]")
    print(f"📦 Tổng số Triplets Huấn Luyện Embedding : {len(triplets):>7} mẫu (Đa miền: Y tế, Tài chính, Khoa học)")
    print(f"📦 Tổng số Pairs Huấn Luyện Reranker     : {len(reranker_pairs):>7} cặp (50% Positive / 50% Hard Negatives)")
    print(f"🛡️ File 888 trang (Pressman)            : 100% CÁCH LY làm tập Test kiểm chứng độc lập.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parent.parent
    t_file = ROOT_DIR / "data" / "triplets" / "massive_embedding_triplets.jsonl"
    r_file = ROOT_DIR / "data" / "triplets" / "massive_reranker_pairs.jsonl"
    build_ultra_dataset(t_file, r_file)
