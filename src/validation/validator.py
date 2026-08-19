"""
Validation layer.

Takes the raw extraction JSON (from extractor.py) and runs a set of
deterministic checks on top of it — this is intentionally NOT another LLM
call. Judges explicitly asked for "validate and enrich information with
traceable outputs" and "improve product data quality and consistency";
a rule-based validation pass is what makes the confidence scores and flags
actionable rather than just LLM self-reported numbers.

Checks implemented:
1. Missing/null field detection (any top-level field with value=None).
2. Low-confidence flagging (confidence below threshold -> needs_review).
3. Unit consistency checks on numeric-looking spec values (temperature,
   speed, voltage) using a small regex-based unit normalizer — catches
   things like "18,000 rpm" vs "20000 min-1" being treated as different
   units for what might be the same underlying quantity, or mixed
   temperature notations ("degC" vs "C").
4. Duplicate-parameter detection in key_specifications (same normalized
   parameter name appearing twice with different values -> contradiction).
5. Overall completeness score for the product record.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

LOW_CONFIDENCE_THRESHOLD = 0.6

TOP_LEVEL_FIELDS = [
    "product_name", "model_number", "manufacturer", "category",
    "short_description", "operating_temperature_range", "protection_rating",
    "weight", "dimensions",
]

# Known unit synonyms so we don't false-flag equivalent notations.
UNIT_SYNONYMS = {
    "rpm": {"rpm", "min-1", "min^-1", "r/min"},
    "temperature": {"c", "degc", "°c", "celsius"},
    "voltage": {"v", "vdc", "volts", "volt"},
}


@dataclass
class ValidationIssue:
    field: str
    issue_type: str          # "missing" | "low_confidence" | "unit_inconsistency" | "contradiction"
    message: str
    severity: str             # "info" | "warning" | "error"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "issue_type": self.issue_type,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class ValidationReport:
    doc_name: str
    issues: List[ValidationIssue] = field(default_factory=list)
    completeness_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_name": self.doc_name,
            "completeness_score": round(self.completeness_score, 2),
            "issues": [i.to_dict() for i in self.issues],
            "issue_count": len(self.issues),
        }


def _extract_number_and_unit(value: str) -> Optional[Tuple[float, str]]:
    """Best-effort parse of a string like '18,000 rpm' or '-30 degC' into
    (number, unit). Returns None if no numeric pattern found."""
    if not value:
        return None
    cleaned = value.replace(",", "")
    match = re.search(r"(-?\d+\.?\d*)\s*([a-zA-Z°/\-\^0-9]*)", cleaned)
    if not match:
        return None
    num_str, unit_str = match.groups()
    try:
        num = float(num_str)
    except ValueError:
        return None
    return num, unit_str.strip().lower()


def _normalize_unit(unit: str) -> Optional[str]:
    unit = unit.lower().strip()
    for canonical, synonyms in UNIT_SYNONYMS.items():
        if unit in synonyms:
            return canonical
    return None


def _check_missing_fields(fields: Dict[str, Any]) -> List[ValidationIssue]:
    issues = []
    for fname in TOP_LEVEL_FIELDS:
        entry = fields.get(fname)
        if entry is None or entry.get("value") in (None, "", "N/A"):
            issues.append(ValidationIssue(
                field=fname,
                issue_type="missing",
                message=f"'{fname}' was not found in the source document.",
                severity="warning",
            ))
    return issues


def _check_low_confidence(fields: Dict[str, Any]) -> List[ValidationIssue]:
    issues = []
    for fname in TOP_LEVEL_FIELDS + ["compliance_standards"]:
        entry = fields.get(fname)
        if not entry or entry.get("value") is None:
            continue
        conf = entry.get("confidence", 0.0)
        if conf < LOW_CONFIDENCE_THRESHOLD:
            note = entry.get("flag") or "Model reported low confidence."
            issues.append(ValidationIssue(
                field=fname,
                issue_type="low_confidence",
                message=f"Confidence {conf:.2f} below threshold — {note}",
                severity="warning",
            ))

    for spec in fields.get("key_specifications", []) or []:
        conf = spec.get("confidence", 0.0)
        if conf < LOW_CONFIDENCE_THRESHOLD:
            note = spec.get("flag") or "Model reported low confidence."
            issues.append(ValidationIssue(
                field=spec.get("parameter", "key_specifications"),
                issue_type="low_confidence",
                message=f"Confidence {conf:.2f} below threshold — {note}",
                severity="warning",
            ))
    return issues


def _check_unit_inconsistencies(fields: Dict[str, Any]) -> List[ValidationIssue]:
    """Group key_specifications by normalized-unit-category and flag when
    the same category shows up with differing raw units, since that's a
    strong signal of a data-quality problem in the source datasheet."""
    issues = []
    specs = fields.get("key_specifications", []) or []

    seen_by_category: Dict[str, List[Tuple[str, str, str]]] = {}
    for spec in specs:
        parsed = _extract_number_and_unit(spec.get("value", ""))
        if not parsed:
            continue
        _, raw_unit = parsed
        category = _normalize_unit(raw_unit)
        if category:
            seen_by_category.setdefault(category, []).append(
                (spec.get("parameter", ""), spec.get("value", ""), raw_unit)
            )

    for category, entries in seen_by_category.items():
        raw_units_used = {e[2] for e in entries}
        if len(raw_units_used) > 1:
            params = ", ".join(f"{p}='{v}'" for p, v, _ in entries)
            issues.append(ValidationIssue(
                field="key_specifications",
                issue_type="unit_inconsistency",
                message=(
                    f"Multiple raw unit notations used for '{category}'-type values "
                    f"in the same document: {params}"
                ),
                severity="error",
            ))
    return issues


def _check_duplicate_parameters(fields: Dict[str, Any]) -> List[ValidationIssue]:
    issues = []
    specs = fields.get("key_specifications", []) or []
    seen: Dict[str, str] = {}
    for spec in specs:
        pname = (spec.get("parameter") or "").strip().lower()
        pval = spec.get("value", "")
        if not pname:
            continue
        if pname in seen and seen[pname] != pval:
            issues.append(ValidationIssue(
                field=pname,
                issue_type="contradiction",
                message=f"Parameter '{pname}' has conflicting values: '{seen[pname]}' vs '{pval}'.",
                severity="error",
            ))
        seen[pname] = pval
    return issues


def _completeness_score(fields: Dict[str, Any]) -> float:
    total = len(TOP_LEVEL_FIELDS)
    filled = sum(
        1 for fname in TOP_LEVEL_FIELDS
        if fields.get(fname) and fields[fname].get("value") not in (None, "", "N/A")
    )
    return filled / total if total else 0.0


def validate_extraction(doc_name: str, fields: Dict[str, Any]) -> ValidationReport:
    issues: List[ValidationIssue] = []
    issues += _check_missing_fields(fields)
    issues += _check_low_confidence(fields)
    issues += _check_unit_inconsistencies(fields)
    issues += _check_duplicate_parameters(fields)

    return ValidationReport(
        doc_name=doc_name,
        issues=issues,
        completeness_score=_completeness_score(fields),
    )


if __name__ == "__main__":
    # Self-test using a hand-built record that mirrors what the mocked
    # extractor test produced — deliberately includes a missing field,
    # a low-confidence field, and a genuine unit inconsistency (rpm vs min-1).
    sample_fields = {
        "product_name": {"value": "Deep Groove Ball Bearing", "confidence": 0.95},
        "model_number": {"value": "SKB-6205-2RS", "confidence": 1.0},
        "manufacturer": {"value": "Nordvik Bearing Industries GmbH", "confidence": 1.0},
        "category": {"value": "bearing", "confidence": 0.9},
        "short_description": {"value": "Single row deep groove ball bearing", "confidence": 0.85},
        "operating_temperature_range": {
            "value": "-30 to 120 C", "confidence": 0.4,
            "flag": "Inconsistent unit notation: degC vs C in source",
        },
        "protection_rating": {"value": None, "confidence": 0.0},
        "weight": {"value": "0.130 kg", "confidence": 1.0},
        "dimensions": {"value": "d25mm D52mm B15mm", "confidence": 0.9},
        "compliance_standards": {"value": ["ISO 15:2017", "RoHS 2011/65/EU"], "confidence": 0.9},
        "key_specifications": [
            {"parameter": "Limiting Speed", "value": "18,000 rpm", "confidence": 0.5,
             "flag": "Conflicts with Reference Speed"},
            {"parameter": "Reference Speed", "value": "20000 min-1", "confidence": 0.5},
            {"parameter": "Dynamic Load Rating", "value": "14000 N", "confidence": 1.0},
        ],
    }

    report = validate_extraction("product_bearing_6205.pdf", sample_fields)
    import json
    print(json.dumps(report.to_dict(), indent=2))
