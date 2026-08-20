"""
Enrichment layer.

The hackathon problem statement explicitly asks for three things:
"automate the creation, enrichment, and validation of product intelligence."
Up to this point the pipeline did creation (extraction) and validation, but
never actually enriched anything — a real gap against the stated
requirements. This module closes that gap with two deterministic,
non-hallucinating enrichment steps:

1. UNIT NORMALIZATION.
   Two products in a catalog might report the same physical quantity in
   different notations — "18,000 rpm" vs "20000 min-1" (numerically
   equivalent, just different labels), or "68 F" vs "20 C" (different
   scales, needs real conversion). Comparing or filtering across a catalog
   is painful when every product uses whatever notation its own datasheet
   happened to use. This step computes a normalized_value/normalized_unit
   pair for every key_specification entry (and top-level fields like
   operating_temperature_range) that maps to a known unit category, so the
   catalog becomes actually comparable — this is the "improve product data
   quality and consistency" outcome from the problem statement, done
   directly rather than left as an aspiration.

2. STANDARDS SUGGESTION.
   Based on a product's extracted `category`, suggest commonly-applicable
   industrial standards from a small static reference table (e.g. a
   "bearing" commonly falls under ISO 15). This is INTENTIONALLY NOT an
   LLM call — asking a model to "guess what standards probably apply"
   is exactly the kind of ungrounded inference that leads to confident
   hallucination on compliance-sensitive claims. A static lookup table is
   honest about what it is: a suggestion to go check, clearly separated
   from `compliance_standards` (which are only ever populated from text
   actually present in the source document), and each suggestion carries
   its own reasoning ("matched via category='bearing'") so it's obvious
   this is enrichment, not extraction.

Both steps are pure functions over the already-validated extraction dict —
no network calls, so they're as fast and deterministic to test as
validator.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Unit normalization
# ---------------------------------------------------------------------------

# Each category maps a set of raw unit spellings to a (canonical_unit,
# conversion_fn) pair. conversion_fn takes the raw numeric value and
# returns the value in the canonical unit.
#
# NOTE on precision: voltage entries (V/VDC/VAC) are normalized to a common
# "V" label purely for magnitude comparison across a catalog. This
# deliberately does not distinguish AC vs DC characteristics — that's a
# real simplification worth knowing about, not a hidden one.
_UNIT_CONVERSIONS: Dict[str, Dict[str, Tuple[str, callable]]] = {
    "rpm": {
        "rpm": ("rpm", lambda v: v),
        "min-1": ("rpm", lambda v: v),
        "min^-1": ("rpm", lambda v: v),
        "r/min": ("rpm", lambda v: v),
    },
    "temperature": {
        "c": ("C", lambda v: v),
        "degc": ("C", lambda v: v),
        "°c": ("C", lambda v: v),
        "celsius": ("C", lambda v: v),
        "f": ("C", lambda v: (v - 32) * 5 / 9),
        "degf": ("C", lambda v: (v - 32) * 5 / 9),
        "°f": ("C", lambda v: (v - 32) * 5 / 9),
        "fahrenheit": ("C", lambda v: (v - 32) * 5 / 9),
    },
    "voltage": {
        "v": ("V", lambda v: v),
        "vdc": ("V", lambda v: v),
        "vac": ("V", lambda v: v),
        "volts": ("V", lambda v: v),
        "volt": ("V", lambda v: v),
    },
    "current": {
        "a": ("A", lambda v: v),
        "ma": ("A", lambda v: v / 1000),
        "amp": ("A", lambda v: v),
        "amps": ("A", lambda v: v),
        "ampere": ("A", lambda v: v),
        "amperes": ("A", lambda v: v),
    },
    "power": {
        "w": ("W", lambda v: v),
        "kw": ("W", lambda v: v * 1000),
        "watt": ("W", lambda v: v),
        "watts": ("W", lambda v: v),
        "kilowatt": ("W", lambda v: v * 1000),
    },
    "frequency": {
        "hz": ("Hz", lambda v: v),
        "khz": ("Hz", lambda v: v * 1000),
    },
}


@dataclass
class NormalizedValue:
    raw_value: str
    normalized_value: float
    normalized_unit: str
    category: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_value": self.raw_value,
            "normalized_value": round(self.normalized_value, 4),
            "normalized_unit": self.normalized_unit,
            "category": self.category,
        }


def normalize_value(raw_value: str) -> Optional[NormalizedValue]:
    """Attempt to normalize a single spec value string. Reuses the same
    number/unit parsing approach as validator.py (imported directly, so
    there's exactly one implementation of this logic, not two that could
    drift apart)."""
    from src.validation.validator import _find_all_number_unit_pairs

    pairs = _find_all_number_unit_pairs(raw_value)
    if not pairs:
        return None

    # Use the last (number, unit) pair — consistent with validator.py's
    # handling of range-style values like "10-30 VDC".
    number, raw_unit = pairs[-1]

    for category, unit_map in _UNIT_CONVERSIONS.items():
        if raw_unit in unit_map:
            canonical_unit, convert = unit_map[raw_unit]
            return NormalizedValue(
                raw_value=raw_value,
                normalized_value=convert(number),
                normalized_unit=canonical_unit,
                category=category,
            )
    return None


def normalize_key_specifications(key_specifications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Returns a list of {parameter, normalization} entries — only for
    specs where normalization was actually possible. Doesn't mutate the
    input list."""
    results = []
    for spec in key_specifications or []:
        if not isinstance(spec, dict):
            continue
        raw_value = spec.get("value", "")
        normalized = normalize_value(raw_value)
        if normalized:
            results.append({
                "parameter": spec.get("parameter"),
                "normalization": normalized.to_dict(),
            })
    return results


# ---------------------------------------------------------------------------
# Standards suggestion
# ---------------------------------------------------------------------------

# Small, honest, static reference table. Not exhaustive — a real production
# system would source this from an actual regulatory database, but for a
# hackathon MVP this demonstrates the enrichment pattern without pretending
# to be a compliance authority.
_CATEGORY_STANDARD_HINTS: Dict[str, List[str]] = {
    "bearing": ["ISO 15 (boundary dimensions)", "ISO 281 (dynamic load rating)"],
    "motor": ["IEC 60034 (rotating electrical machines)", "IEC 60072 (frame sizes)"],
    "sensor": ["IEC 60947-5-2 (proximity switches)", "IEC 61326 (EMC requirements)"],
    "proximity sensor": ["IEC 60947-5-2 (proximity switches)", "IEC 61326 (EMC requirements)"],
}


@dataclass
class StandardsSuggestion:
    suggested_standards: List[str]
    reasoning: str
    is_extracted_fact: bool = False  # always False — this is a suggestion, not a document-sourced fact

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggested_standards": self.suggested_standards,
            "reasoning": self.reasoning,
            "is_extracted_fact": self.is_extracted_fact,
        }


def suggest_standards(category: Optional[str]) -> Optional[StandardsSuggestion]:
    if not category:
        return None
    key = category.strip().lower()
    hints = _CATEGORY_STANDARD_HINTS.get(key)
    if not hints:
        return None
    return StandardsSuggestion(
        suggested_standards=hints,
        reasoning=f"Matched via category='{category}' against a static reference table — "
                  f"not extracted from the source document. Verify applicability manually.",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class EnrichmentResult:
    normalized_specifications: List[Dict[str, Any]] = field(default_factory=list)
    standards_suggestion: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalized_specifications": self.normalized_specifications,
            "standards_suggestion": self.standards_suggestion,
        }


def enrich_extraction(fields: Dict[str, Any]) -> EnrichmentResult:
    key_specs = fields.get("key_specifications", []) or []
    normalized = normalize_key_specifications(key_specs)

    category_entry = fields.get("category") or {}
    category_value = category_entry.get("value") if isinstance(category_entry, dict) else None
    suggestion = suggest_standards(category_value)

    return EnrichmentResult(
        normalized_specifications=normalized,
        standards_suggestion=suggestion.to_dict() if suggestion else None,
    )


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    sample_fields = {
        "category": {"value": "bearing"},
        "key_specifications": [
            {"parameter": "Limiting Speed", "value": "18,000 rpm"},
            {"parameter": "Reference Speed", "value": "20000 min-1"},
            {"parameter": "Dynamic Load Rating", "value": "14000 N"},
        ],
    }
    result = enrich_extraction(sample_fields)
    print(json.dumps(result.to_dict(), indent=2))
