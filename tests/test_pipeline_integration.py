import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.pdf_loader import load_pdf, Chunk
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
    fake_response.parsed = None  # force the manual json.loads + validate fallback path
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


def test_pipeline_handles_large_document_via_batched_extraction(mock_extractor, tmp_path):
    """End-to-end check that a document large enough to need multiple
    extraction batches still produces one coherent merged product record
    through the full ingest -> extract -> validate -> save chain."""
    # Build a synthetic "document" with enough chunk volume to force batching.
    big_chunks = [
        Chunk(chunk_id=f"big-p{i}-text", doc_id="bigdoc", doc_name="big_catalog_entry.pdf",
              page_number=i, text=f"Page {i} content. " * 400, kind="text")
        for i in range(1, 11)
    ]
    from src.ingestion.pdf_loader import ParsedDocument
    big_doc = ParsedDocument(
        doc_id="bigdoc", doc_name="big_catalog_entry.pdf",
        file_path="/fake/path.pdf", num_pages=10, chunks=big_chunks,
    )

    from src.extraction.extractor import _batch_chunks
    batches = _batch_chunks(big_doc.chunks)
    assert len(batches) > 1, "test setup should actually exercise multi-batch behavior"

    call_log = []

    def side_effect(*args, **kwargs):
        call_log.append(1)
        resp = MagicMock()
        resp.parsed = None
        # Each batch "finds" a different field, to prove merging actually happens
        idx = len(call_log)
        payload = _mock_json(big_doc)
        payload["model_number"]["value"] = f"BATCH-{idx}-MODEL"
        payload["model_number"]["confidence"] = 0.5 + (idx * 0.05)  # later batches "more confident"
        resp.text = json.dumps(payload)
        return resp

    mock_extractor.client.models.generate_content.side_effect = side_effect

    record = process_document(big_doc, mock_extractor)

    assert len(call_log) == len(batches), "should make exactly one API call per batch"
    # The merge logic should have kept the highest-confidence model_number
    assert record.extraction["model_number"]["value"] == f"BATCH-{len(batches)}-MODEL"

    saved_path = record.save(output_dir=tmp_path)
    assert saved_path.exists()
