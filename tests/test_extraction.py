import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.ingestion.pdf_loader import Chunk, load_pdf
from src.extraction.extractor import (
    GeminiExtractor,
    ProductExtraction,
    FieldExtraction,
    ComplianceExtraction,
    KeySpecExtraction,
    _format_chunks_block,
    _batch_chunks,
    _merge_extractions,
    MAX_CHARS_PER_BATCH,
)
from google.genai.errors import APIError

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
    with patch.object(GeminiExtractor, "__init__", lambda self, **kw: None):
        extractor = GeminiExtractor.__new__(GeminiExtractor)
        extractor.model = "gemini-2.5-flash"
        extractor.client = MagicMock()
        yield extractor


def _make_extraction(model_number="MOCK-1", confidence=0.9) -> ProductExtraction:
    return ProductExtraction(
        product_name=FieldExtraction(value="Mock Product", confidence=confidence),
        model_number=FieldExtraction(value=model_number, confidence=confidence),
        compliance_standards=ComplianceExtraction(value=["CE"], confidence=confidence),
        key_specifications=[
            KeySpecExtraction(parameter="Voltage", value="24 V", source_chunk_id="c1",
                               source_page=1, confidence=confidence)
        ],
    )


# ---------------------------------------------------------------------------
# Chunk formatting / batching
# ---------------------------------------------------------------------------

def test_format_chunks_block_includes_all_chunks(bearing_doc):
    block = _format_chunks_block(bearing_doc.chunks)
    for chunk in bearing_doc.chunks:
        assert chunk.chunk_id in block
        assert f"page={chunk.page_number}" in block


def test_batch_chunks_stays_under_budget():
    chunks = [
        Chunk(chunk_id=f"d-p{i}", doc_id="d", doc_name="x.pdf", page_number=i,
              text="x" * 2000, kind="text")
        for i in range(1, 21)
    ]
    batches = _batch_chunks(chunks)
    for batch in batches:
        total = sum(len(c.text) for c in batch)
        assert total <= MAX_CHARS_PER_BATCH or len(batch) == 1


def test_batch_chunks_preserves_all_chunks_no_loss_no_duplication():
    chunks = [
        Chunk(chunk_id=f"d-p{i}", doc_id="d", doc_name="x.pdf", page_number=i,
              text="x" * 2000, kind="text")
        for i in range(1, 21)
    ]
    batches = _batch_chunks(chunks)
    ids_in = {c.chunk_id for c in chunks}
    ids_out = {c.chunk_id for b in batches for c in b}
    assert ids_in == ids_out


def test_batch_chunks_handles_oversized_single_chunk():
    huge = Chunk(chunk_id="huge", doc_id="d", doc_name="x.pdf", page_number=1,
                 text="y" * 50_000, kind="text")
    batches = _batch_chunks([huge])
    assert len(batches) == 1
    assert len(batches[0]) == 1  # not dropped, not split mid-chunk


def test_batch_chunks_small_doc_is_single_batch(bearing_doc):
    # Our sample datasheets are well under the char budget individually.
    batches = _batch_chunks(bearing_doc.chunks)
    assert len(batches) == 1


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def test_merge_extractions_single_result_passthrough():
    r = _make_extraction()
    assert _merge_extractions([r]) is r


def test_merge_extractions_prefers_higher_confidence():
    r1 = ProductExtraction(product_name=FieldExtraction(value="Widget", confidence=0.5))
    r2 = ProductExtraction(product_name=FieldExtraction(value="Widget Pro", confidence=0.95))
    merged = _merge_extractions([r1, r2])
    assert merged.product_name.value == "Widget Pro"


def test_merge_extractions_fills_gaps_across_batches():
    r1 = ProductExtraction(model_number=FieldExtraction(value=None, confidence=0.0))
    r2 = ProductExtraction(model_number=FieldExtraction(value="WP-100", confidence=0.9))
    merged = _merge_extractions([r1, r2])
    assert merged.model_number.value == "WP-100"


def test_merge_extractions_unions_compliance_standards():
    r1 = ProductExtraction(compliance_standards=ComplianceExtraction(value=["CE"], confidence=0.8))
    r2 = ProductExtraction(compliance_standards=ComplianceExtraction(value=["RoHS"], confidence=0.7))
    merged = _merge_extractions([r1, r2])
    assert set(merged.compliance_standards.value) == {"CE", "RoHS"}


def test_merge_extractions_dedupes_key_specifications_by_parameter():
    r1 = ProductExtraction(key_specifications=[
        KeySpecExtraction(parameter="Voltage", value="24V", source_chunk_id="c1",
                           source_page=1, confidence=0.6),
    ])
    r2 = ProductExtraction(key_specifications=[
        KeySpecExtraction(parameter="voltage", value="24V", source_chunk_id="c2",
                           source_page=2, confidence=0.9),
        KeySpecExtraction(parameter="Current", value="2A", source_chunk_id="c2",
                           source_page=2, confidence=0.9),
    ])
    merged = _merge_extractions([r1, r2])
    assert len(merged.key_specifications) == 2  # Voltage deduped (case-insensitive), Current added
    voltage_spec = next(s for s in merged.key_specifications if s.parameter.lower() == "voltage")
    assert voltage_spec.confidence == 0.9  # higher-confidence version kept


# ---------------------------------------------------------------------------
# Schema validation (the core hardening fix)
# ---------------------------------------------------------------------------

def test_extract_accepts_sdk_auto_parsed_response(mock_extractor, bearing_doc):
    """When the SDK successfully auto-parses via response_schema, we should
    use response.parsed directly rather than re-parsing response.text."""
    mock_extraction = _make_extraction()
    fake_response = MagicMock()
    fake_response.parsed = mock_extraction
    mock_extractor.client.models.generate_content.return_value = fake_response

    result = mock_extractor.extract(bearing_doc)
    assert result.raw_json["model_number"]["value"] == "MOCK-1"


def test_extract_falls_back_to_manual_parse_when_not_auto_parsed(mock_extractor, bearing_doc):
    """If response.parsed isn't a ProductExtraction (e.g. SDK couldn't
    auto-parse), we must fall back to json.loads + explicit validation."""
    valid_json = json.dumps(_make_extraction().model_dump())
    fake_response = MagicMock()
    fake_response.parsed = None
    fake_response.text = valid_json
    mock_extractor.client.models.generate_content.return_value = fake_response

    result = mock_extractor.extract(bearing_doc)
    assert result.raw_json["model_number"]["value"] == "MOCK-1"


def test_extract_raises_clear_error_on_invalid_json(mock_extractor, bearing_doc):
    fake_response = MagicMock()
    fake_response.parsed = None
    fake_response.text = "not valid json {{{"
    mock_extractor.client.models.generate_content.return_value = fake_response

    with pytest.raises(ValueError, match="did not return valid JSON"):
        mock_extractor.extract(bearing_doc)


def test_extract_raises_clear_error_on_schema_mismatch(mock_extractor, bearing_doc):
    """This is the regression test for the core hardening fix: a
    structurally-valid JSON response that violates the schema (wrong type
    for confidence) must fail loudly and clearly, not propagate bad data
    downstream into validator.py or the UI."""
    fake_response = MagicMock()
    fake_response.parsed = None
    fake_response.text = json.dumps({"model_number": {"confidence": "not-a-float!!"}})
    mock_extractor.client.models.generate_content.return_value = fake_response

    with pytest.raises(ValueError, match="did not match the expected schema"):
        mock_extractor.extract(bearing_doc)


def test_schema_validation_failures_are_not_retried(mock_extractor, bearing_doc):
    """Retrying a deterministic schema mismatch wastes API quota for no
    benefit — it will fail the same way every time."""
    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        resp = MagicMock()
        resp.parsed = None
        resp.text = json.dumps({"model_number": {"confidence": "bad"}})
        return resp

    mock_extractor.client.models.generate_content.side_effect = side_effect

    with pytest.raises(ValueError):
        mock_extractor.extract(bearing_doc)

    assert call_count["n"] == 1


def test_extract_batch_continues_after_individual_failure(mock_extractor, bearing_doc):
    fake_response = MagicMock()
    fake_response.parsed = None
    fake_response.text = "not valid json"
    mock_extractor.client.models.generate_content.return_value = fake_response

    results = mock_extractor.extract_batch([bearing_doc, bearing_doc])
    assert results == []


def test_missing_api_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiExtractor()


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

def test_transient_api_error_is_retried_and_recovers(mock_extractor):
    call_count = {"n": 0}
    valid_json = json.dumps(_make_extraction().model_dump())

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise APIError(code=503, response_json={"error": "transient"})
        resp = MagicMock()
        resp.parsed = None
        resp.text = valid_json
        return resp

    mock_extractor.client.models.generate_content.side_effect = side_effect

    response = mock_extractor._call_gemini("dummy prompt")
    assert call_count["n"] == 3
    assert response.text == valid_json


def test_retry_gives_up_after_max_attempts(mock_extractor):
    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        raise APIError(code=503, response_json={"error": "persistent failure"})

    mock_extractor.client.models.generate_content.side_effect = side_effect

    with pytest.raises(APIError):
        mock_extractor._call_gemini("dummy prompt")

    assert call_count["n"] == 3  # stop_after_attempt(3)
