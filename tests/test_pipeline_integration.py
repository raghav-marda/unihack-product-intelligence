import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.pdf_loader import load_pdf
from src.extraction.extractor import GeminiExtractor
from src.pipeline import process_document, REPO_ROOT, OUTPUT_DIR

SAMPLES_DIR = REPO_ROOT / "data" / "samples"
BEARING_PDF = SAMPLES_DIR / "product_bearing_6205.pdf"


def test_repo_root_resolves_correctly():
    """Regression test: REPO_ROOT must point at the actual repo root
    (was previously off by one directory level)."""
    assert REPO_ROOT.name == "unihack-product-intelligence"
    assert (REPO_ROOT / "requirements.txt").exists()
    assert OUTPUT_DIR == REPO_ROOT / "data" / "output"


@pytest.fixture
def mock_extractor():
    with patch.object(GeminiExtractor, "__init__", lambda self, **kw: None):
        extractor = GeminiExtractor.__new__(GeminiExtractor)
        extractor.model = "gemini-2.5-flash"
        extractor.client = MagicMock()
        yield extractor


def _mock_json(doc):
    return {
        "product_name": {"value": "Mock Product", "source_chunk_id": doc.chunks[0].chunk_id,
                          "source_page": 1, "confidence": 0.9, "flag": None},
        "model_number": {"value": "MOCK-1", "source_chunk_id": doc.chunks[0].chunk_id,
                          "source_page": 1, "confidence": 0.9, "flag": None},
        "manufacturer": {"value": "Mock Mfr", "source_chunk_id": doc.chunks[0].chunk_id,
                          "source_page": 1, "confidence": 0.9, "flag": None},
        "category": {"value": "generic", "source_chunk_id": doc.chunks[0].chunk_id,
                      "source_page": 1, "confidence": 0.8, "flag": None},
        "short_description": {"value": "desc", "source_chunk_id": doc.chunks[0].chunk_id,
                               "source_page": 1, "confidence": 0.8, "flag": None},
        "operating_temperature_range": {"value": None, "source_chunk_id": None,
                                         "source_page": None, "confidence": 0.0, "flag": None},
        "protection_rating": {"value": "IP55", "source_chunk_id": doc.chunks[0].chunk_id,
                               "source_page": 1, "confidence": 0.9, "flag": None},
        "weight": {"value": "1 kg", "source_chunk_id": doc.chunks[0].chunk_id,
                   "source_page": 1, "confidence": 0.9, "flag": None},
        "dimensions": {"value": "10x10x10 mm", "source_chunk_id": doc.chunks[0].chunk_id,
                       "source_page": 1, "confidence": 0.9, "flag": None},
        "compliance_standards": {"value": ["CE"], "source_chunk_id": doc.chunks[0].chunk_id,
                                  "source_page": 1, "confidence": 0.9, "flag": None},
        "key_specifications": [
            {"parameter": "Voltage", "value": "24 V", "source_chunk_id": doc.chunks[0].chunk_id,
             "source_page": 1, "confidence": 0.9, "flag": None},
        ],
    }


def test_full_pipeline_end_to_end(mock_extractor, tmp_path, monkeypatch):
    if not BEARING_PDF.exists():
        pytest.skip("Sample PDF not found")

    doc = load_pdf(str(BEARING_PDF))
    fake_response = MagicMock()
    fake_response.text = json.dumps(_mock_json(doc))
    mock_extractor.client.models.generate_content.return_value = fake_response

    record = process_document(doc, mock_extractor)

    assert record.doc_id == doc.doc_id
    assert record.extraction["product_name"]["value"] == "Mock Product"
    assert "completeness_score" in record.validation
    assert record.validation["issue_count"] >= 1  # operating_temperature_range is missing

    # Save to a temp dir instead of the real output dir, to avoid polluting it
    saved_path = record.save(output_dir=tmp_path)
    assert saved_path.exists()
    reloaded = json.loads(saved_path.read_text())
    assert reloaded["doc_name"] == doc.doc_name
    assert reloaded["extraction"]["model_number"]["value"] == "MOCK-1"
