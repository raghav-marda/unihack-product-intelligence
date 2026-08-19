import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.pdf_loader import load_pdf
from src.extraction.extractor import GeminiExtractor, _format_chunks_block

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
BEARING_PDF = SAMPLES_DIR / "product_bearing_6205.pdf"


@pytest.fixture
def bearing_doc():
    if not BEARING_PDF.exists():
        pytest.skip("Sample PDF not found — run scripts/generate_sample_data.py first")
    return load_pdf(str(BEARING_PDF))


@pytest.fixture
def mock_extractor():
    """A GeminiExtractor whose network client is fully mocked, so tests never
    touch the real API."""
    with patch.object(GeminiExtractor, "__init__", lambda self, **kw: None):
        extractor = GeminiExtractor.__new__(GeminiExtractor)
        extractor.model = "gemini-2.5-flash"
        extractor.client = MagicMock()
        yield extractor


def _valid_mock_json(doc):
    return {
        "product_name": {"value": "Deep Groove Ball Bearing", "source_chunk_id": doc.chunks[0].chunk_id,
                          "source_page": 1, "confidence": 0.95, "flag": None},
        "model_number": {"value": "SKB-6205-2RS", "source_chunk_id": doc.chunks[0].chunk_id,
                          "source_page": 1, "confidence": 1.0, "flag": None},
        "manufacturer": {"value": "Nordvik Bearing Industries GmbH", "source_chunk_id": doc.chunks[0].chunk_id,
                          "source_page": 1, "confidence": 1.0, "flag": None},
        "category": {"value": "bearing", "source_chunk_id": doc.chunks[0].chunk_id,
                      "source_page": 1, "confidence": 0.9, "flag": None},
        "short_description": {"value": "Single row deep groove ball bearing", "source_chunk_id": doc.chunks[0].chunk_id,
                               "source_page": 1, "confidence": 0.85, "flag": None},
        "operating_temperature_range": {"value": "-30 to 120 C", "source_chunk_id": doc.chunks[0].chunk_id,
                                         "source_page": 1, "confidence": 0.4,
                                         "flag": "Inconsistent unit notation"},
        "protection_rating": {"value": None, "source_chunk_id": None, "source_page": None,
                               "confidence": 0.0, "flag": None},
        "weight": {"value": "0.130 kg", "source_chunk_id": doc.chunks[0].chunk_id,
                   "source_page": 1, "confidence": 1.0, "flag": None},
        "dimensions": {"value": "d25mm D52mm B15mm", "source_chunk_id": doc.chunks[0].chunk_id,
                       "source_page": 1, "confidence": 0.9, "flag": None},
        "compliance_standards": {"value": ["ISO 15:2017", "RoHS 2011/65/EU"], "source_chunk_id": doc.chunks[0].chunk_id,
                                  "source_page": 2, "confidence": 0.9, "flag": None},
        "key_specifications": [
            {"parameter": "Dynamic Load Rating", "value": "14000 N", "source_chunk_id": doc.chunks[0].chunk_id,
             "source_page": 1, "confidence": 1.0, "flag": None},
        ],
    }


def test_format_chunks_block_includes_all_chunks(bearing_doc):
    block = _format_chunks_block(bearing_doc)
    for chunk in bearing_doc.chunks:
        assert chunk.chunk_id in block
        assert f"page={chunk.page_number}" in block


def test_extract_parses_valid_json_response(mock_extractor, bearing_doc):
    mock_json = _valid_mock_json(bearing_doc)
    fake_response = MagicMock()
    fake_response.text = json.dumps(mock_json)
    mock_extractor.client.models.generate_content.return_value = fake_response

    result = mock_extractor.extract(bearing_doc)

    assert result.doc_id == bearing_doc.doc_id
    assert result.doc_name == bearing_doc.doc_name
    assert result.raw_json["model_number"]["value"] == "SKB-6205-2RS"
    assert result.raw_json["protection_rating"]["value"] is None


def test_extract_raises_on_invalid_json(mock_extractor, bearing_doc):
    fake_response = MagicMock()
    fake_response.text = "not valid json {{{"
    mock_extractor.client.models.generate_content.return_value = fake_response

    with pytest.raises(ValueError, match="did not return valid JSON"):
        mock_extractor.extract(bearing_doc)


def test_extract_batch_continues_after_individual_failure(mock_extractor, bearing_doc):
    fake_response = MagicMock()
    fake_response.text = "not valid json"
    mock_extractor.client.models.generate_content.return_value = fake_response

    results = mock_extractor.extract_batch([bearing_doc, bearing_doc])
    assert results == []  # both failed, but no exception propagated


def test_missing_api_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiExtractor()
