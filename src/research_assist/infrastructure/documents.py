"""Bounded extraction and chunking. Physical PDF pages are preserved, never guessed."""

import io
import re
import unicodedata
import zipfile
from pathlib import PurePath
from typing import Any, Literal, cast
from uuid import uuid4

from research_assist.config import Settings
from research_assist.domain.errors import DocumentError
from research_assist.domain.models import Chunk, Document, SourceSection

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def safe_filename(filename: str) -> str:
    basename = PurePath(filename.replace("\\", "/")).name
    basename = re.sub(r"[\x00-\x1f\x7f]", "", basename).strip().strip(".")
    if not basename or len(basename) > 180:
        raise DocumentError("Tên tệp không hợp lệ (tối đa 180 ký tự).")
    return basename


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract(data: bytes, filename: str, settings: Settings) -> list[SourceSection]:
    suffix = PurePath(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise DocumentError("Chỉ hỗ trợ PDF, DOCX, TXT và Markdown.")
    if not data or len(data) > settings.max_upload_mb * 1024 * 1024:
        raise DocumentError(f"Tệp rỗng hoặc lớn hơn {settings.max_upload_mb} MB.")
    sections: list[SourceSection] = []
    try:
        if suffix == ".pdf":
            import pymupdf

            if not data.lstrip().startswith(b"%PDF-"):
                raise DocumentError("Tệp không có định dạng PDF hợp lệ.")
            with cast(Any, pymupdf.open)(stream=data, filetype="pdf") as pdf:
                if pdf.needs_pass:
                    raise DocumentError("PDF có mật khẩu. Hãy tải bản đã mở khóa.")
                if len(pdf) > settings.max_pages:
                    raise DocumentError(f"PDF vượt quá {settings.max_pages} trang.")
                total = 0
                ocr_pages = 0
                for index, page in enumerate(pdf):
                    text = normalize_text(page.get_text("text", sort=True))
                    extraction: Literal["text", "ocr"] = "text"
                    warnings: list[str] = []
                    if len(text) < 40 and page.get_images():
                        if settings.pdf_ocr_enabled:
                            ocr_pages += 1
                            if ocr_pages > settings.pdf_ocr_max_pages:
                                raise DocumentError(
                                    "PDF vượt số trang OCR cho phép. Hãy chia nhỏ tài liệu."
                                )
                            pixels = (
                                page.rect.width
                                * page.rect.height
                                * (settings.pdf_ocr_dpi / 72) ** 2
                            )
                            if pixels > 20_000_000:
                                raise DocumentError(
                                    "Trang OCR quá lớn. Giảm RA_PDF_OCR_DPI hoặc chia trang."
                                )
                            try:
                                textpage = page.get_textpage_ocr(
                                    language=settings.pdf_ocr_language,
                                    dpi=settings.pdf_ocr_dpi,
                                    full=True,
                                )
                                text = normalize_text(
                                    page.get_text("text", textpage=textpage, sort=True)
                                )
                                extraction = "ocr"
                                warnings.append(
                                    f"Trang {index + 1}: văn bản OCR cần đối chiếu ảnh gốc, "
                                    "nhất là số liệu."
                                )
                            except Exception as exc:
                                raise DocumentError(
                                    "Không chạy được OCR. Kiểm tra Tesseract/tessdata "
                                    "và RA_PDF_OCR_LANGUAGE."
                                ) from exc
                        else:
                            warnings.append(
                                f"Trang {index + 1}: ít văn bản và có ảnh; có thể cần OCR."
                            )
                    elif len(text) < 40:
                        warnings.append(
                            f"Trang {index + 1}: ít hoặc không có văn bản để lập chỉ mục."
                        )
                    total += len(text)
                    if total > settings.max_document_chars:
                        raise DocumentError("Tài liệu vượt giới hạn văn bản cho một lần tải lên.")
                    sections.append(
                        SourceSection(
                            text=text,
                            page=index + 1,
                            label=f"Trang {index + 1}" + (" · OCR" if extraction == "ocr" else ""),
                            extraction=extraction,
                            warnings=warnings,
                        )
                    )
        elif suffix == ".docx":
            from docx import Document as WordDocument

            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                infos = archive.infolist()
                if len(infos) > 1500 or sum(i.file_size for i in infos) > 24 * 1024 * 1024:
                    raise DocumentError("DOCX có dữ liệu giải nén quá lớn.")
                if "word/document.xml" not in archive.namelist():
                    raise DocumentError("Tệp không phải DOCX hợp lệ.")
            word = cast(Any, WordDocument)(io.BytesIO(data))
            # XML flow order keeps paragraphs and tables in their original order.
            from docx.oxml.ns import qn
            from docx.table import Table
            from docx.text.paragraph import Paragraph

            parts = []
            for child in word.element.body:
                if child.tag == qn("w:p"):
                    parts.append(Paragraph(child, word).text)
                elif child.tag == qn("w:tbl"):
                    table = Table(child, word)
                    parts.extend(" | ".join(cell.text for cell in row.cells) for row in table.rows)
            sections = [SourceSection(text=normalize_text("\n".join(parts)), label="Văn bản DOCX")]
        else:
            sections = [
                SourceSection(text=normalize_text(data.decode("utf-8-sig")), label="Văn bản")
            ]
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("Không đọc được tệp. Hãy kiểm tra định dạng và mã hóa UTF-8.") from exc
    if sum(len(s.text) for s in sections) > settings.max_document_chars:
        raise DocumentError("Tài liệu vượt giới hạn văn bản cho một lần tải lên.")
    if sum(len(s.text) for s in sections) < 40:
        raise DocumentError("Không đủ văn bản để lập chỉ mục. PDF scan cần OCR trước khi tải lên.")
    return sections


def chunk_sections(
    sections: list[SourceSection], document: Document, settings: Settings
) -> list[Chunk]:
    if settings.chunk_strategy == "tokens":
        from research_assist.infrastructure.token_chunker import token_chunks

        return token_chunks(sections, document, settings)
    chunks: list[Chunk] = []
    for section in sections:
        start = 0
        while start < len(section.text):
            end = min(start + settings.chunk_chars, len(section.text))
            if end < len(section.text):
                # Prefer a paragraph/sentence/space boundary without creating tiny fragments.
                lower = start + settings.chunk_chars // 2
                boundary = max(section.text.rfind(sep, lower, end) for sep in ("\n", ". ", " "))
                if boundary > lower:
                    end = boundary + 1
            raw = section.text[start:end]
            left_trim = len(raw) - len(raw.lstrip())
            text = raw.strip()
            if len(text) >= 20:
                offset = start + left_trim
                chunks.append(
                    Chunk(
                        id=f"ch_{uuid4().hex}",
                        document_id=document.id,
                        document_title=document.title,
                        filename=document.filename,
                        text=text,
                        page=section.page,
                        section=section.label,
                        ordinal=len(chunks),
                        start=offset,
                        end=offset + len(text),
                    )
                )
            if end >= len(section.text):
                break
            start = max(start + 1, end - settings.chunk_overlap)
    return chunks
