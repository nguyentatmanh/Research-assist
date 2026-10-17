import pymupdf
import pytest

from research_assist.config import Settings
from research_assist.domain.errors import DocumentError
from research_assist.infrastructure.documents import extract


def image_pdf() -> bytes:
    with pymupdf.open() as text_pdf, pymupdf.open() as scan:
        page = text_pdf.new_page()
        page.insert_text(
            (50, 100), "Retrieval uses evidence from scientific documents.", fontsize=20
        )
        image = page.get_pixmap(dpi=100).tobytes("png")
        target = scan.new_page()
        target.insert_image(target.rect, stream=image)
        return scan.tobytes()


def test_scan_without_ocr_fails_with_actionable_error(settings: Settings) -> None:
    with pytest.raises(DocumentError, match="OCR"):
        extract(image_pdf(), "scan.pdf", settings)


def test_ocr_failure_is_explicit(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    settings.pdf_ocr_enabled = True

    def failed(*args, **kwargs):
        raise RuntimeError("no tessdata")

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", failed)
    with pytest.raises(DocumentError, match="Tesseract"):
        extract(image_pdf(), "scan.pdf", settings)


def test_ocr_provenance_and_one_based_page_are_preserved(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.pdf_ocr_enabled = True

    def simulated_ocr(page, **kwargs):
        page.insert_text(
            (50, 100), "Retrieval uses evidence from scientific documents.", fontsize=20
        )
        return page.get_textpage()

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", simulated_ocr)
    sections = extract(image_pdf(), "scan.pdf", settings)
    assert sections[0].extraction == "ocr" and sections[0].page == 1
    assert "OCR" in sections[0].label and sections[0].warnings
