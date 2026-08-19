from src.validation.validator import (
    validate_extraction,
    _extract_number_and_unit,
    _find_all_number_unit_pairs,
    LOW_CONFIDENCE_THRESHOLD,
)


def _base_fields(**overrides):
    fields = {
        "product_name": {"value": "Test Product", "confidence": 0.9},
        "model_number": {"value": "TP-100", "confidence": 0.9},
        "manufacturer": {"value": "Test Mfr", "confidence": 0.9},
        "category": {"value": "generic", "confidence": 0.9},
        "short_description": {"value": "A test product", "confidence": 0.9},
        "operating_temperature_range": {"value": "-20 to 60 C", "confidence": 0.9},
        "protection_rating": {"value": "IP65", "confidence": 0.9},
        "weight": {"value": "2 kg", "confidence": 0.9},
        "dimensions": {"value": "10x10x10 mm", "confidence": 0.9},
        "compliance_standards": {"value": ["CE"], "confidence": 0.9},
        "key_specifications": [],
    }
    fields.update(overrides)
    return fields


# ---------------------------------------------------------------------------
# Regression tests: the number/unit regex bug (real bug found in review)
#
# v1's regex used a permissive unit character class that included digits
# and bare hyphens. For a range value like "10-30 VDC" — a REAL field in
# our own sensor sample datasheet (Operating Voltage) — it parsed the
# "unit" as "-30" instead of "vdc", meaning any actual voltage-unit
# inconsistency in a real document would have been silently missed.
# ---------------------------------------------------------------------------

def test_range_value_parses_correct_unit_not_the_second_number():
    result = _extract_number_and_unit("10-30 VDC")
    assert result is not None
    _, unit = result
    assert unit == "vdc", f"BUG REGRESSION: got unit={unit!r}, expected 'vdc'"


def test_range_value_with_percent_still_parses_voltage_unit():
    result = _extract_number_and_unit("415 V (+/-10%)")
    assert result[1] == "v"


def test_min_inverse_notation_parses_as_single_unit_token():
    result = _extract_number_and_unit("20000 min-1")
    assert result[1] == "min-1"


def test_simple_values_still_parse_correctly():
    cases = {
        "18,000 rpm": "rpm",
        "8 mm (Sn, non-flush)": "mm",
        "200 mA": "ma",
        "500 Hz": "hz",
        "0.130 kg": "kg",
    }
    for value, expected_unit in cases.items():
        result = _extract_number_and_unit(value)
        assert result is not None, f"Failed to parse: {value!r}"
        assert result[1] == expected_unit, f"{value!r} -> {result[1]!r}, expected {expected_unit!r}"


def test_find_all_pairs_returns_multiple_matches():
    pairs = _find_all_number_unit_pairs("-30 degC to 120 C")
    units = [u for _, u in pairs]
    assert "degc" in units
    assert "c" in units
    assert len(pairs) == 2


def test_empty_or_none_value_returns_none():
    assert _extract_number_and_unit("") is None
    assert _extract_number_and_unit(None) is None


# ---------------------------------------------------------------------------
# Missing field / low confidence checks
# ---------------------------------------------------------------------------

def test_fully_complete_record_has_no_missing_issues():
    fields = _base_fields()
    report = validate_extraction("test.pdf", fields)
    missing = [i for i in report.issues if i.issue_type == "missing"]
    assert missing == []
    assert report.completeness_score == 1.0


def test_missing_field_is_flagged():
    fields = _base_fields(protection_rating={"value": None, "confidence": 0.0})
    report = validate_extraction("test.pdf", fields)
    missing_fields = {i.field for i in report.issues if i.issue_type == "missing"}
    assert "protection_rating" in missing_fields
    assert report.completeness_score < 1.0


def test_missing_check_does_not_crash_on_malformed_entry():
    """Defensive: if an entry isn't even a dict (malformed upstream data),
    the checker should flag it as missing, not raise an AttributeError."""
    fields = _base_fields(protection_rating="not a dict")
    report = validate_extraction("test.pdf", fields)
    assert any(i.field == "protection_rating" for i in report.issues)


def test_low_confidence_field_is_flagged():
    fields = _base_fields(
        weight={"value": "2 kg", "confidence": LOW_CONFIDENCE_THRESHOLD - 0.1}
    )
    report = validate_extraction("test.pdf", fields)
    low_conf = [i for i in report.issues if i.issue_type == "low_confidence"]
    assert any(i.field == "weight" for i in low_conf)


# ---------------------------------------------------------------------------
# Cross-field unit inconsistency (across different key_specifications)
# ---------------------------------------------------------------------------

def test_unit_inconsistency_detected_for_rpm_vs_min1():
    fields = _base_fields(key_specifications=[
        {"parameter": "Limiting Speed", "value": "18,000 rpm", "confidence": 0.9},
        {"parameter": "Reference Speed", "value": "20000 min-1", "confidence": 0.9},
    ])
    report = validate_extraction("test.pdf", fields)
    unit_issues = [i for i in report.issues if i.issue_type == "unit_inconsistency"
                   and i.field == "key_specifications"]
    assert len(unit_issues) == 1
    assert unit_issues[0].severity == "error"


def test_unit_inconsistency_detected_for_range_voltage_vs_plain_voltage():
    """Regression test tied directly to the regex fix: before the fix,
    '10-30 VDC' would never have been categorized as voltage at all, so
    this inconsistency would have been silently missed."""
    fields = _base_fields(key_specifications=[
        {"parameter": "Operating Voltage", "value": "10-30 VDC", "confidence": 0.9},
        {"parameter": "Rated Voltage", "value": "24 V", "confidence": 0.9},
    ])
    report = validate_extraction("test.pdf", fields)
    unit_issues = [i for i in report.issues if i.issue_type == "unit_inconsistency"
                   and i.field == "key_specifications"]
    assert len(unit_issues) == 1
    assert "voltage" in unit_issues[0].message.lower()


def test_no_unit_inconsistency_when_units_match():
    fields = _base_fields(key_specifications=[
        {"parameter": "Speed A", "value": "1000 rpm", "confidence": 0.9},
        {"parameter": "Speed B", "value": "2000 rpm", "confidence": 0.9},
    ])
    report = validate_extraction("test.pdf", fields)
    unit_issues = [i for i in report.issues if i.issue_type == "unit_inconsistency"
                   and i.field == "key_specifications"]
    assert unit_issues == []


# ---------------------------------------------------------------------------
# NEW: internal single-field unit mixing (independent of LLM self-report)
# ---------------------------------------------------------------------------

def test_internal_field_unit_mixing_detected_without_llm_flag():
    """This is the core regression test for the new check: the field's
    own confidence is high (0.9) and it has NO self-reported 'flag' from
    the model — the validator must catch the degC/C mixing purely from
    the value string itself."""
    fields = _base_fields(
        operating_temperature_range={"value": "-30 degC to 120 C", "confidence": 0.9}
        # deliberately no "flag" key here
    )
    report = validate_extraction("test.pdf", fields)
    internal_issues = [i for i in report.issues if i.issue_type == "unit_inconsistency"
                        and i.field == "operating_temperature_range"]
    assert len(internal_issues) == 1
    assert "temperature" in internal_issues[0].message.lower()


def test_no_internal_mixing_flagged_when_units_are_consistent():
    fields = _base_fields(
        operating_temperature_range={"value": "-20 C to 60 C", "confidence": 0.9}
    )
    report = validate_extraction("test.pdf", fields)
    internal_issues = [i for i in report.issues if i.issue_type == "unit_inconsistency"
                        and i.field == "operating_temperature_range"]
    assert internal_issues == []


def test_internal_mixing_check_skips_non_string_values():
    fields = _base_fields(operating_temperature_range={"value": None, "confidence": 0.0})
    report = validate_extraction("test.pdf", fields)
    # Should not crash, and the "missing" check (not unit_inconsistency) is what fires
    internal_issues = [i for i in report.issues if i.issue_type == "unit_inconsistency"
                        and i.field == "operating_temperature_range"]
    assert internal_issues == []


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

def test_contradiction_detected_for_duplicate_parameter_different_values():
    fields = _base_fields(key_specifications=[
        {"parameter": "Voltage", "value": "24V", "confidence": 0.9},
        {"parameter": "voltage", "value": "12V", "confidence": 0.9},
    ])
    report = validate_extraction("test.pdf", fields)
    contradictions = [i for i in report.issues if i.issue_type == "contradiction"]
    assert len(contradictions) == 1


def test_no_contradiction_for_duplicate_parameter_same_value():
    fields = _base_fields(key_specifications=[
        {"parameter": "Voltage", "value": "24V", "confidence": 0.9},
        {"parameter": "Voltage", "value": "24V", "confidence": 0.9},
    ])
    report = validate_extraction("test.pdf", fields)
    contradictions = [i for i in report.issues if i.issue_type == "contradiction"]
    assert contradictions == []


# ---------------------------------------------------------------------------
# Completeness score
# ---------------------------------------------------------------------------

def test_completeness_score_bounds():
    empty_fields = _base_fields(**{
        k: {"value": None, "confidence": 0.0}
        for k in ["product_name", "model_number", "manufacturer", "category",
                   "short_description", "operating_temperature_range",
                   "protection_rating", "weight", "dimensions"]
    })
    report = validate_extraction("test.pdf", empty_fields)
    assert report.completeness_score == 0.0


def test_full_validate_extraction_end_to_end_matches_real_bearing_scenario():
    """Integration-style test mirroring our actual bearing sample datasheet's
    known data-quality issues, to lock in expected validator behavior."""
    fields = _base_fields(
        protection_rating={"value": None, "confidence": 0.0},
        operating_temperature_range={"value": "-30 degC to 120 C", "confidence": 0.7},
        key_specifications=[
            {"parameter": "Limiting Speed", "value": "18,000 rpm", "confidence": 0.9},
            {"parameter": "Reference Speed", "value": "20000 min-1", "confidence": 0.9},
            {"parameter": "Dynamic Load Rating", "value": "14000 N", "confidence": 1.0},
        ],
    )
    report = validate_extraction("product_bearing_6205.pdf", fields)
    issue_types = {(i.field, i.issue_type) for i in report.issues}
    assert ("protection_rating", "missing") in issue_types
    assert ("key_specifications", "unit_inconsistency") in issue_types
    assert ("operating_temperature_range", "unit_inconsistency") in issue_types
    assert len(report.issues) == 3
