from src.validation.validator import validate_extraction, LOW_CONFIDENCE_THRESHOLD


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


def test_low_confidence_field_is_flagged():
    fields = _base_fields(
        weight={"value": "2 kg", "confidence": LOW_CONFIDENCE_THRESHOLD - 0.1}
    )
    report = validate_extraction("test.pdf", fields)
    low_conf = [i for i in report.issues if i.issue_type == "low_confidence"]
    assert any(i.field == "weight" for i in low_conf)


def test_unit_inconsistency_detected_for_rpm_vs_min1():
    fields = _base_fields(key_specifications=[
        {"parameter": "Limiting Speed", "value": "18,000 rpm", "confidence": 0.9},
        {"parameter": "Reference Speed", "value": "20000 min-1", "confidence": 0.9},
    ])
    report = validate_extraction("test.pdf", fields)
    unit_issues = [i for i in report.issues if i.issue_type == "unit_inconsistency"]
    assert len(unit_issues) == 1
    assert unit_issues[0].severity == "error"


def test_no_unit_inconsistency_when_units_match():
    fields = _base_fields(key_specifications=[
        {"parameter": "Speed A", "value": "1000 rpm", "confidence": 0.9},
        {"parameter": "Speed B", "value": "2000 rpm", "confidence": 0.9},
    ])
    report = validate_extraction("test.pdf", fields)
    unit_issues = [i for i in report.issues if i.issue_type == "unit_inconsistency"]
    assert unit_issues == []


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


def test_completeness_score_bounds():
    empty_fields = _base_fields(**{
        k: {"value": None, "confidence": 0.0}
        for k in ["product_name", "model_number", "manufacturer", "category",
                   "short_description", "operating_temperature_range",
                   "protection_rating", "weight", "dimensions"]
    })
    report = validate_extraction("test.pdf", empty_fields)
    assert report.completeness_score == 0.0
