import pymupdf
import pytest
from fastapi.testclient import TestClient

from research_assist.agents.reader import ReaderAgent
from research_assist.api.app import create_app
from research_assist.domain.errors import DocumentError
from research_assist.domain.models import ReaderOutput, RetrievedChunk
from research_assist.infrastructure.documents import chunk_sections, extract
from research_assist.infrastructure.llm import DemoLanguageModel
from research_assist.infrastructure.source_preview import render_region


def two_columns():
    pdf = pymupdf.open()
    page = pdf.new_page(width=600, height=800)
    page.insert_text((50, 40), "A research paper with two columns", fontsize=16)
    # Deliberately write in interleaved order; reading must finish the left column first.
    for y, text in [
        (100, "LEFT_A describes the proposed research method."),
        (170, "LEFT_B completes the method and its conditions."),
    ]:
        page.insert_text((40, y), text, fontsize=9)
        page.insert_text((325, y), text.replace("LEFT_", "RIGHT_"), fontsize=9)
    return pdf


def test_columns_are_contiguous_with_exact_provenance(settings, document, tmp_path):
    with two_columns() as pdf:
        data = pdf.tobytes()
    sections = extract(data, "columns.pdf", settings)
    text = sections[0].text
    assert text.index("LEFT_A") < text.index("LEFT_B") < text.index("RIGHT_A")
    assert text.count("LEFT_A") == text.count("RIGHT_B") == 1
    chunks = chunk_sections(sections, document, settings)
    for chunk in chunks:
        assert chunk.text == text[chunk.start : chunk.end]
        assert chunk.regions
        assert all(chunk.start <= r.start < r.end <= chunk.end for r in chunk.regions)
    retrieved = RetrievedChunk(**chunks[0].model_dump(), score=1.0)
    quote = "LEFT_A describes the proposed research method."
    evidence, _, _ = ReaderAgent(DemoLanguageModel()).validate_output(
        "What is the method?",
        [retrieved],
        ReaderOutput.model_validate(
            {
                "evidence": [
                    {
                        "chunk_id": retrieved.id,
                        "quote": quote,
                        "relevance": "Exact source passage.",
                    }
                ]
            }
        ),
    )
    assert len(evidence) == 1 and evidence[0].regions
    path = tmp_path / "columns.pdf"
    path.write_bytes(data)
    clip = ",".join(str(v) for v in evidence[0].regions[0].bbox)
    assert render_region(path, 1, clip).startswith(b"\x89PNG")
    assert path.read_bytes() == data


@pytest.mark.parametrize(
    "clip,page", [("nan,0,100,100", 1), ("0,0,9000,100", 1), ("20,20,10,30", 1), ("0,0,100,100", 2)]
)
def test_source_preview_rejects_invalid_coordinates(tmp_path, clip, page):
    path = tmp_path / "source.pdf"
    with two_columns() as pdf:
        pdf.save(path)
    with pytest.raises(DocumentError):
        render_region(path, page, clip)


def test_ruled_table_keeps_rows_and_no_duplicate_cells(settings):
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=400)
        for x in (40, 160, 280):
            page.draw_line((x, 60), (x, 150))
        for y in (60, 90, 120, 150):
            page.draw_line((40, y), (280, y))
        for row, cells in enumerate(
            [("Model", "Accuracy"), ("Encoder", "95%"), ("Decoder", "87%")]
        ):
            for column, cell in enumerate(cells):
                page.insert_text((50 + column * 120, 80 + row * 30), cell, fontsize=10)
        data = pdf.tobytes()
    text = extract(data, "table.pdf", settings)[0].text
    assert "Encoder | 95%" in text and "Decoder | 87%" in text
    assert text.count("95%") == 1


def test_source_region_http_route_validates_page_and_clip(settings):
    with two_columns() as pdf:
        data = pdf.tobytes()
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/documents", files={"file": ("columns.pdf", data, "application/pdf")}
        )
        doc = response.json()["document"]
        root = f"/api/documents/{doc['id']}/pages"
        preview = client.get(root + "/1/region", params={"clip": "30,80,250,180"})
        assert preview.status_code == 200 and preview.headers["content-type"] == "image/png"
        assert preview.content.startswith(b"\x89PNG")
        assert client.get(root + "/2/region", params={"clip": "0,0,100,100"}).status_code == 400
        assert client.get(root + "/1/region", params={"clip": "nan,0,100,100"}).status_code == 400
