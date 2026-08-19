"""
Validation layer (v2 — hardened).

Takes the raw extraction JSON (from extractor.py) and runs a set of
deterministic checks on top of it. This is intentionally NOT another LLM
call — a rule-based pass is what makes the confidence scores and flags
actionable rather than just LLM self-reported numbers.

What changed from v1, and why:

1. FIXED: number/unit parsing broke on range values.
   The v1 regex used a permissive unit character class that included
   digits and bare hyphens, so a value like "10-30 VDC" (a real field in
   our own sensor sample datasheet) parsed as unit="-30" instead of
   unit="vdc" — meaning any real inconsistency involving a range-style
   voltage/current spec would have been silently missed. The unit group
   now requires starting with a letter (or "°"), and we scan ALL matches
   in the string via finditer rather than trusting a single greedy
   search() from the start — so a range's trailing unit is found instead
   of getting swallowed by the leading number.

2. NEW: internal single-field unit-mixing check.
   v1 only checked for unit inconsistency ACROSS different
   key_specifications entries. It never independently checked whether a
   single field's own value — like operating_temperature_range containing
   "-30 degC to 120 C" — mixes two notations of the same unit within one
   string. That case was previously only caught if the LLM happened to
   self-report it via the "flag" field, which defeats the point of having
   a deterministic validator at all. Now checked directly with regex,
   independent of what the model says.

3. Expanded unit-category coverage (current, power, frequency) so the
   cross-field consistency check isn't limited to just rpm/temperature/
   voltage — real datasheets (including our own sensor sample) use mA, Hz,
   W routinely.
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
    "temperature": {"c", "degc", "°c", "celsius", "f", "degf", "°f", "fahrenheit"},
    "voltage": {"v", "vdc", "vac", "volts", "volt"},
    "current": {"a", "ma", "amp", "amps", "ampere", "amperes"},
    "power": {"w", "kw", "watt", "watts", "kilowatt"},
    "frequency": {"hz", "khz"},
}

# Unit token must start with a letter or a degree symbol, so a bare
# leading "-30" (from a range like "10-30 VDC") is never mistaken for a
# unit. "min-1"-style suffixes are still supported via the optional
# trailing "-<digits>" group.
_NUM_UNIT_RE = re.compile(r"(-?\d+\.?\d*)\s*(°?[a-zA-Z/]+(?:-\d+)?)")


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


def _normalize_unit(unit: str) -> Optional[str]:
    unit = unit.lower().strip()
    for canonical, synonyms in UNIT_SYNONYMS.items():
        if unit in synonyms:
            return canonical
    return None


def _find_all_number_unit_pairs(value: str) -> List[Tuple[float, str]]:
    """Find every (number, raw_unit) occurrence in a string. Returns raw
    (non-normalized) units so callers can report the exact source text."""
    if not value:
        return []
    cleaned = value.replace(",", "")
    pairs = []
    for m in _NUM_UNIT_RE.finditer(cleaned):
        num_str, unit_str = m.groups()
        try:
            num = float(num_str)
        except ValueError:
            continue
        pairs.append((num, unit_str.strip().lower()))
    return pairs


def _extract_number_and_unit(value: str) -> Optional[Tuple[float, str]]:
    """Best-effort single (number, unit) pair — prefers the LAST match in
    the string, since for range values like '10-30 VDC' the trailing unit
    (found near the end) is the one that actually matters for
    categorization; the leading number in a range is not meaningfully
    "the" value on its own anyway."""
    pairs = _find_all_number_unit_pairs(value)
    return pairs[-1] if pairs else None


def _check_missing_fields(fields: Dict[str, Any]) -> List[ValidationIssue]:
    issues = []
    for fname in TOP_LEVEL_FIELDS:
        entry = fields.get(fname)
        if not isinstance(entry, dict) or entry.get("value") in (None, "", "N/A"):
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
        if not isinstance(entry, dict) or entry.get("value") is None:
            continue
        conf = entry.get("confidence", 0.0) or 0.0
        if conf < LOW_CONFIDENCE_THRESHOLD:
            note = entry.get("flag") or "Model reported low confidence."
            issues.append(ValidationIssue(
                field=fname,
                issue_type="low_confidence",
                message=f"Confidence {conf:.2f} below threshold — {note}",
                severity="warning",
            ))

    for spec in fields.get("key_specifications", []) or []:
        if not isinstance(spec, dict):
            continue
        conf = spec.get("confidence", 0.0) or 0.0
        if conf < LOW_CONFIDENCE_THRESHOLD:
            note = spec.get("flag") or "Model reported low confidence."
            issues.append(ValidationIssue(
                field=spec.get("parameter", "key_specifications"),
                issue_type="low_confidence",
                message=f"Confidence {conf:.2f} below threshold — {note}",
                severity="warning",
            ))
    return issues


def _check_cross_field_unit_inconsistencies(fields: Dict[str, Any]) -> List[ValidationIssue]:
    """Group key_specifications by normalized-unit-category and flag when
    the same category shows up with differing raw units, since that's a
    strong signal of a data-quality problem in the source datasheet."""
    issues = []
    specs = fields.get("key_specifications", []) or []

    seen_by_category: Dict[str, List[Tuple[str, str, str]]] = {}
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        parsed = _extract_number_and_unit(spec.get("value", "") or "")
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


def _check_internal_field_unit_mixing(fields: Dict[str, Any]) -> List[ValidationIssue]:
    """Check whether a SINGLE field's own string value mixes two different
    raw unit notations of the same category — e.g.
    operating_temperature_range = "-30 degC to 120 C" uses both "degC" and
    "C" for temperature within one string. This is checked independently
    of the model's self-reported 'flag', so it doesn't rely on the LLM
    happening to notice."""
    issues = []
    for fname in TOP_LEVEL_FIELDS:
        entry = fields.get(fname)
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        if not isinstance(value, str) or not value:
            continue

        pairs = _find_all_number_unit_pairs(value)
        by_category: Dict[str, set] = {}
        for _, raw_unit in pairs:
            category = _normalize_unit(raw_unit)
            if category:
                by_category.setdefault(category, set()).add(raw_unit)

        for category, raw_units in by_category.items():
            if len(raw_units) > 1:
                issues.append(ValidationIssue(
                    field=fname,
                    issue_type="unit_inconsistency",
                    message=(
                        f"'{fname}' mixes multiple raw unit notations for "
                        f"'{category}'-type values within a single value: "
                        f"{sorted(raw_units)} (source text: '{value}')"
                    ),
                    severity="error",
                ))
    return issues


def _check_duplicate_parameters(fields: Dict[str, Any]) -> List[ValidationIssue]:
    issues = []
    specs = fields.get("key_specifications", []) or []
    seen: Dict[str, str] = {}
    for spec in specs:
        if not isinstance(spec, dict):
            continue
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
        if isinstance(fields.get(fname), dict)
        and fields[fname].get("value") not in (None, "", "N/A")
    )
    return filled / total if total else 0.0


def validate_extraction(doc_name: str, fields: Dict[str, Any]) -> ValidationReport:
    issues: List[ValidationIssue] = []
    issues += _check_missing_fields(fields)
    issues += _check_low_confidence(fields)
    issues += _check_cross_field_unit_inconsistencies(fields)
    issues += _check_internal_field_unit_mixing(fields)
    issues += _check_duplicate_parameters(fields)

    return ValidationReport(
        doc_name=doc_name,
        issues=issues,
        completeness_score=_completeness_score(fields),
    )


if __name__ == "__main__":
    # Self-test with a record that mirrors real extraction output from our
    # sample bearing datasheet — includes a missing field, a genuine
    # cross-field unit inconsistency (rpm vs min-1), AND an internal
    # single-field mixing issue (temperature range with two notations)
    # that v1's validator would have missed entirely without an LLM flag.
    sample_fields = {
        "product_name": {"value": "Deep Groove Ball Bearing", "confidence": 0.95},
        "model_number": {"value": "SKB-6205-2RS", "confidence": 1.0},
        "manufacturer": {"value": "Nordvik Bearing Industries GmbH", "confidence": 1.0},
        "category": {"value": "bearing", "confidence": 0.9},
        "short_description": {"value": "Single row deep groove ball bearing", "confidence": 0.85},
        "operating_temperature_range": {
            "value": "-30 degC to 120 C", "confidence": 0.7,
        },
        "protection_rating": {"value": None, "confidence": 0.0},
        "weight": {"value": "0.130 kg", "confidence": 1.0},
        "dimensions": {"value": "d25mm D52mm B15mm", "confidence": 0.9},
        "compliance_standards": {"value": ["ISO 15:2017", "RoHS 2011/65/EU"], "confidence": 0.9},
        "key_specifications": [
            {"parameter": "Limiting Speed", "value": "18,000 rpm", "confidence": 0.9},
            {"parameter": "Reference Speed", "value": "20000 min-1", "confidence": 0.9},
            {"parameter": "Dynamic Load Rating", "value": "14000 N", "confidence": 1.0},
        ],
    }

    report = validate_extraction("product_bearing_6205.pdf", sample_fields)
    import json
    print(json.dumps(report.to_dict(), indent=2))
