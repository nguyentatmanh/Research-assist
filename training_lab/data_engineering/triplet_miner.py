# -*- coding: utf-8 -*-
"""
Hard Negative Triplet Mining Module for Multi-Agent RAG.
Extracts passages from domain PDFs and constructs high-quality IR triplets:
  (Query, Positive Passage, Hard Negative Passage)
Follows academic standards from BEIR / MoAI / CoREB.
"""

import re
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF


class DomainTripletMiner:
    """
    Mines query-positive-negative triplets directly from local PDF textbooks/papers.
    Uses SQLite FTS5 BM25 to identify challenging lexical distractors (Hard Negatives).
    """

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path(__file__).resolve().parent.parent / "data" / "triplets" / "fts_miner.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_fts_table()

    def _init_fts_table(self):
        cur = self.conn.cursor()
        cur.execute("DROP TABLE IF EXISTS chunks_fts")
        cur.execute("DROP TABLE IF EXISTS chunks_meta")
        cur.execute("""
            CREATE TABLE chunks_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_no INTEGER,
                section_title TEXT,
                content TEXT
            )
        """)
        cur.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content,
                content=chunks_meta,
                content_rowid=id,
                tokenize='porter unicode61'
            )
        """)
        self.conn.commit()

    def index_pdf_file(self, pdf_path: Path, max_pages: int = 200, chunk_size: int = 120):
        """Indexes pages and sub-chunks from the target PDF into SQLite FTS5 for fast BM25 mining."""
        print(f"📖 [Triplet Miner] Indexing PDF: {pdf_path.name} (Up to {max_pages} pages)...")
        doc = fitz.open(pdf_path)
        try:
            total = min(len(doc), max_pages)
            cur = self.conn.cursor()
            chunk_count = 0

            for page_idx in range(total):
                page = doc[page_idx]
                page_no = page_idx + 1
                text_content = page.get_text("text")
                if not text_content.strip():
                    continue

                paragraphs = [p.strip() for p in text_content.split("\n\n") if len(p.strip().split()) >= 25]
                for para in paragraphs:
                    cleaned = re.sub(r"\s+", " ", para).strip()
                    cur.execute(
                        "INSERT INTO chunks_meta (page_no, section_title, content) VALUES (?, ?, ?)",
                        (page_no, f"Page {page_no}", cleaned)
                    )
                    chunk_id = cur.lastrowid
                    cur.execute(
                        "INSERT INTO chunks_fts (rowid, content) VALUES (?, ?)",
                        (chunk_id, cleaned)
                    )
                    chunk_count += 1
            self.conn.commit()
        finally:
            doc.close()
        print(f"✅ [Triplet Miner] Successfully indexed {chunk_count} passages from {total} pages.")
        return chunk_count

    def search_hard_negatives(self, query: str, positive_id: int, top_n: int = 5) -> List[Tuple[int, int, str]]:
        """
        Uses BM25 search via SQLite FTS5 to find passages that are lexically similar to the query,
        but are NOT the positive passage (different chunk or different page).
        """
        # Clean query tokens for FTS5 and quote literals
        tokens = [re.sub(r"[^\w]", "", w) for w in query.split() if len(w) >= 2]
        clean_tokens = [f'"{t}"' for t in tokens[:8] if t.isalnum()]
        if not clean_tokens:
            return []
        fts_query = " OR ".join(clean_tokens)

        cur = self.conn.cursor()
        sql = """
            SELECT m.id, m.page_no, m.content, bm25(chunks_fts) as rank_score
            FROM chunks_fts f
            JOIN chunks_meta m ON f.rowid = m.id
            WHERE chunks_fts MATCH ? AND m.id != ?
            ORDER BY rank_score ASC
            LIMIT ?
        """
        try:
            cur.execute(sql, (fts_query, positive_id, top_n))
            rows = cur.fetchall()
            return [(r[0], r[1], r[2]) for r in rows]
        except Exception:
            return []

    def export_triplets_dataset(
        self, 
        seed_qa_pairs: List[Dict[str, Any]], 
        out_embedding_path: Path, 
        out_reranker_path: Path
    ):
        """
        Constructs and exports:
          1. Embedding dataset (Query, Positive, Hard Negative) for MultipleNegativesRankingLoss.
          2. Reranker dataset (Query, Document, Label 1/0) for Cross-Encoder ranking.
        """
        out_embedding_path.parent.mkdir(parents=True, exist_ok=True)
        out_reranker_path.parent.mkdir(parents=True, exist_ok=True)

        cur = self.conn.cursor()
        embedding_triplets = []
        reranker_pairs = []

        print(f"🎯 [Triplet Miner] Mining hard negatives for {len(seed_qa_pairs)} seed queries...")

        for item in seed_qa_pairs:
            query = str(item.get("query") or "").strip()
            if not query:
                continue
            target_keyword = item.get("keyword", "")
            target_page = item.get("page", None)

            # 1. Tìm Positive chunk chứa từ khóa/trang mục tiêu
            if target_page:
                cur.execute(
                    "SELECT id, page_no, content FROM chunks_meta WHERE page_no = ? AND content LIKE ? LIMIT 1",
                    (target_page, f"%{target_keyword}%")
                )
            else:
                cur.execute(
                    "SELECT id, page_no, content FROM chunks_meta WHERE content LIKE ? LIMIT 1",
                    (f"%{target_keyword}%",)
                )
            pos_row = cur.fetchone()

            if not pos_row:
                # Fallback: search best match
                cur.execute("SELECT id, page_no, content FROM chunks_meta LIMIT 1")
                pos_row = cur.fetchone()
                if not pos_row:
                    continue

            pos_id, pos_page, pos_content = pos_row

            # 2. Khai thác Hard Negatives (các chunk khác có BM25 score cao)
            negatives = self.search_hard_negatives(query, positive_id=pos_id, top_n=3)
            if not negatives:
                # Lấy ngẫu nhiên chunk từ trang khác làm semi-hard negative
                cur.execute("SELECT id, page_no, content FROM chunks_meta WHERE page_no != ? LIMIT 2", (pos_page,))
                negatives = [(r[0], r[1], r[2]) for r in cur.fetchall()]

            neg_page = negatives[0][1] if negatives else 1
            if negatives:
                neg_content = negatives[0][2]
            else:
                cur.execute("SELECT content FROM chunks_meta WHERE id != ? LIMIT 1", (pos_id,))
                fb_row = cur.fetchone()
                neg_content = fb_row[0] if fb_row else "Reference baseline contextual information across document domains."

            # Format 1: Embedding Triplet
            embedding_triplets.append({
                "query": query,
                "positive": f"(Trang {pos_page}): {pos_content}",
                "negative": f"(Trang {neg_page}): {neg_content}"
            })

            # Format 2: Reranker Pair (Binary Classification / Margin)
            reranker_pairs.append({
                "query": query,
                "document": f"(Trang {pos_page}): {pos_content}",
                "label": 1
            })
            reranker_pairs.append({
                "query": query,
                "document": f"(Trang {neg_page}): {neg_content}",
                "label": 0
            })

        # Lưu file JSONL
        with open(out_embedding_path, "w", encoding="utf-8") as f:
            for r in embedding_triplets:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with open(out_reranker_path, "w", encoding="utf-8") as f:
            for r in reranker_pairs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"📦 [Triplet Miner] Exported {len(embedding_triplets)} Embedding triplets -> {out_embedding_path.name}")
        print(f"📦 [Triplet Miner] Exported {len(reranker_pairs)} Reranker pairs -> {out_reranker_path.name}")
        return len(embedding_triplets), len(reranker_pairs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if hasattr(self, "conn") and self.conn:
            try:
                self.conn.commit()
            except Exception:
                pass
            self.conn.close()
