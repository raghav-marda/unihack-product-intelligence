from src.enrichment.enricher import (
    normalize_value,
    normalize_key_specifications,
    suggest_standards,
    enrich_extraction,
)


# ---------------------------------------------------------------------------
# Unit normalization — correctness of the actual conversion math
# ---------------------------------------------------------------------------

def test_rpm_and_min_inverse_normalize_to_same_canonical_unit():
    """This is the whole point of normalization: two datasheet notations
    for the same physical quantity should become directly comparable."""
    a = normalize_value("18,000 rpm")
    b = normalize_value("20000 min-1")
    assert a.normalized_unit == b.normalized_unit == "rpm"
    assert a.normalized_value == 18000.0
    assert b.normalized_value == 20000.0


def test_fahrenheit_converts_correctly_to_celsius():
    result = normalize_value("68 F")
    assert result.normalized_unit == "C"
    assert abs(result.normalized_value - 20.0) < 0.001


def test_fahrenheit_boiling_and_freezing_points():
    freezing = normalize_value("32 F")
    boiling = normalize_value("212 F")
    assert abs(freezing.normalized_value - 0.0) < 0.001
    assert abs(boiling.normalized_value - 100.0) < 0.001


def test_celsius_passes_through_unchanged():
    result = normalize_value("120 C")
    assert result.normalized_unit == "C"
    assert result.normalized_value == 120.0


def test_milliamps_converts_to_amps():
    result = normalize_value("500 mA")
    assert result.normalized_unit == "A"
    assert abs(result.normalized_value - 0.5) < 0.0001


def test_kilowatts_converts_to_watts():
    result = normalize_value("2 kW")
    assert result.normalized_unit == "W"
    assert result.normalized_value == 2000.0


def test_kilohertz_converts_to_hertz():
    result = normalize_value("1.5 kHz")
    assert result.normalized_unit == "Hz"
    assert result.normalized_value == 1500.0


def test_voltage_variants_normalize_to_common_unit():
    a = normalize_value("415 V")
    b = normalize_value("24 VDC")
    assert a.normalized_unit == b.normalized_unit == "V"


def test_unrecognized_unit_returns_none():
    result = normalize_value("some text with no unit")
    assert result is None


def test_empty_value_returns_none():
    assert normalize_value("") is None
    assert normalize_value(None) is None


def test_normalize_key_specifications_only_includes_normalizable_entries():
    specs = [
        {"parameter": "Limiting Speed", "value": "18,000 rpm"},
        {"parameter": "Model Name", "value": "not a numeric unit at all"},
        {"parameter": "Weight", "value": "0.130 kg"},  # kg not in our unit tables — skipped, honestly
    ]
    result = normalize_key_specifications(specs)
    params = {r["parameter"] for r in result}
    assert "Limiting Speed" in params
    assert "Model Name" not in params


def test_normalize_key_specifications_handles_malformed_entries_gracefully():
    specs = [{"parameter": "X"}, "not even a dict", None, {"value": "18000 rpm"}]
    result = normalize_key_specifications(specs)  # should not raise
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Standards suggestion — must be clearly separated from extracted facts
# ---------------------------------------------------------------------------

def test_bearing_category_suggests_iso_standards():
    suggestion = suggest_standards("bearing")
    assert suggestion is not None
    assert any("ISO" in s for s in suggestion.suggested_standards)


def test_suggestion_is_explicitly_marked_as_not_extracted():
    """Critical: a suggestion must never be confusable with a fact actually
    found in the source document — that would be a hallucination risk
    disguised as extraction."""
    suggestion = suggest_standards("motor")
    assert suggestion.is_extracted_fact is False
    assert "not extracted" in suggestion.reasoning.lower()


def test_unknown_category_returns_no_suggestion_rather_than_guessing():
    assert suggest_standards("some totally novel product type xyz") is None


def test_none_category_returns_no_suggestion():
    assert suggest_standards(None) is None


def test_category_matching_is_case_insensitive():
    a = suggest_standards("Bearing")
    b = suggest_standards("bearing")
    assert a.suggested_standards == b.suggested_standards


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------

def test_enrich_extraction_end_to_end():
    fields = {
        "category": {"value": "bearing"},
        "key_specifications": [
            {"parameter": "Limiting Speed", "value": "18,000 rpm"},
            {"parameter": "Reference Speed", "value": "20000 min-1"},
        ],
    }
    result = enrich_extraction(fields)
    assert len(result.normalized_specifications) == 2
    assert result.standards_suggestion is not None
    assert result.standards_suggestion["is_extracted_fact"] is False


def test_enrich_extraction_handles_missing_category_and_specs():
    result = enrich_extraction({})
    assert result.normalized_specifications == []
    assert result.standards_suggestion is None
