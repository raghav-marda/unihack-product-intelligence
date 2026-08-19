import os
from pathlib import Path

import pytest

from src.ingestion.pdf_loader import load_pdf, load_pdfs_from_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "data" / "samples"


@pytest.fixture(scope="module")
def sample_docs():
    if not SAMPLES_DIR.exists() or not any(SAMPLES_DIR.glob("*.pdf")):
        pytest.skip("No sample PDFs found — run scripts/generate_sample_data.py first")
    return load_pdfs_from_dir(str(SAMPLES_DIR))


def test_loads_all_sample_pdfs(sample_docs):
    assert len(sample_docs) == 3
    names = {d.doc_name for d in sample_docs}
    assert "product_bearing_6205.pdf" in names
    assert "product_motor_im100l.pdf" in names
    assert "product_sensor_ips18.pdf" in names


def test_each_doc_has_chunks(sample_docs):
    for doc in sample_docs:
        assert len(doc.chunks) > 0, f"{doc.doc_name} produced no chunks"


def test_chunks_have_page_and_id_metadata(sample_docs):
    for doc in sample_docs:
        for chunk in doc.chunks:
            assert chunk.page_number >= 1
            assert chunk.chunk_id
            assert chunk.doc_id == doc.doc_id
            assert chunk.kind in ("text", "table")


def test_table_chunks_are_extracted(sample_docs):
    """Every sample datasheet has spec tables — make sure at least one
    table-kind chunk was found per document (this is the data that matters
    most for extraction)."""
    for doc in sample_docs:
        table_chunks = [c for c in doc.chunks if c.kind == "table"]
        assert len(table_chunks) > 0, f"{doc.doc_name} produced no table chunks"


def test_doc_id_is_stable_across_reloads():
    bearing_path = SAMPLES_DIR / "product_bearing_6205.pdf"
    if not bearing_path.exists():
        pytest.skip("Sample PDF not found")
    doc1 = load_pdf(str(bearing_path))
    doc2 = load_pdf(str(bearing_path))
    assert doc1.doc_id == doc2.doc_id


def test_load_pdf_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pdf("/nonexistent/path/to/file.pdf")


def test_full_text_concatenates_chunks(sample_docs):
    doc = sample_docs[0]
    full = doc.full_text()
    assert len(full) > 0
    assert doc.chunks[0].text in full
