"""
UI tests using Streamlit's AppTest framework — this actually executes
src/ui/app.py's script logic and simulates real widget interactions
(typing into a text_input and triggering a rerun), rather than just unit
testing isolated helper functions. This is what caught the
compliance_standards list-corruption bug: a plain unit test of a helper
function would never have exercised the actual Streamlit rerun path where
the bug lived.
"""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "output"
APP_PATH = REPO_ROOT / "src" / "ui" / "app.py"


def _write_fixture_record(doc_name: str, compliance_value=("CE", "RoHS")) -> Path:
    record = {
        "doc_id": "testdoc123",
        "doc_name": doc_name,
        "extraction": {
            "product_name": {"value": "Test Widget", "confidence": 0.9,
                              "source_page": 1, "source_chunk_id": "c1", "flag": None},
            "model_number": {"value": "TW-1", "confidence": 0.9,
                              "source_page": 1, "source_chunk_id": "c1", "flag": None},
            "manufacturer": {"value": None, "confidence": 0.0,
                              "source_page": None, "source_chunk_id": None, "flag": None},
            "category": {"value": None, "confidence": 0.0,
                         "source_page": None, "source_chunk_id": None, "flag": None},
            "short_description": {"value": None, "confidence": 0.0,
                                   "source_page": None, "source_chunk_id": None, "flag": None},
            "operating_temperature_range": {"value": None, "confidence": 0.0,
                                             "source_page": None, "source_chunk_id": None, "flag": None},
            "protection_rating": {"value": None, "confidence": 0.0,
                                   "source_page": None, "source_chunk_id": None, "flag": None},
            "weight": {"value": None, "confidence": 0.0,
                       "source_page": None, "source_chunk_id": None, "flag": None},
            "dimensions": {"value": None, "confidence": 0.0,
                           "source_page": None, "source_chunk_id": None, "flag": None},
            "compliance_standards": {"value": list(compliance_value), "confidence": 0.9,
                                      "source_page": 1, "source_chunk_id": "c1", "flag": None},
            "key_specifications": [],
        },
        "validation": {"completeness_score": 0.2, "issues": [], "issue_count": 0},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"testdoc123_{Path(doc_name).stem}.json"
    path.write_text(json.dumps(record))
    return path


@pytest.fixture
def fixture_record():
    doc_name = "test_product_ui.pdf"
    path = _write_fixture_record(doc_name)
    yield doc_name
    path.unlink(missing_ok=True)


def test_app_runs_without_exception(fixture_record):
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    assert not at.exception


def test_editing_list_field_preserves_list_type(fixture_record):
    """Regression test for the compliance_standards type-corruption bug:
    editing the text_input for a list-valued field must write the value
    back as a list, not silently downgrade it to a comma-joined string."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    assert not at.exception

    rec_before = at.session_state.records[fixture_record]
    assert rec_before["extraction"]["compliance_standards"]["value"] == ["CE", "RoHS"]

    target_key = f"{fixture_record}::compliance_standards"
    matches = [w for w in at.text_input if w.key == target_key]
    assert len(matches) == 1, "expected exactly one text_input widget for this field"

    matches[0].set_value("CE, RoHS, UKCA").run()
    assert not at.exception

    rec_after = at.session_state.records[fixture_record]
    value_after = rec_after["extraction"]["compliance_standards"]["value"]
    assert isinstance(value_after, list), (
        f"BUG REGRESSION: compliance_standards became {type(value_after).__name__}, "
        f"expected list"
    )
    assert value_after == ["CE", "RoHS", "UKCA"]


def test_editing_scalar_field_still_works_as_plain_string(fixture_record):
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)

    target_key = f"{fixture_record}::model_number"
    matches = [w for w in at.text_input if w.key == target_key]
    assert len(matches) == 1

    matches[0].set_value("TW-2-REVISED").run()
    assert not at.exception

    rec_after = at.session_state.records[fixture_record]
    assert rec_after["extraction"]["model_number"]["value"] == "TW-2-REVISED"
