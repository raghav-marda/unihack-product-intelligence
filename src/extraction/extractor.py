"""
Extraction agent.

Takes a ParsedDocument (from pdf_loader) and calls Gemini to produce a
structured product-intelligence record: a fixed schema of attribute fields,
each tagged with the page/chunk it was sourced from (traceability) and a
confidence score. This is the core "AI-powered" step of the pipeline.

Design notes:
- We feed the model the FULL set of chunks for one product (page-tagged),
  and ask for strict JSON output matching PRODUCT_SCHEMA.
- Every extracted field must include: value, source_chunk_id, source_page,
  confidence (0-1). If a field isn't found in the document, the model must
  return it as null with confidence 0 rather than guessing — this is
  enforced via the prompt AND re-checked in validation.py.
- We use response_mime_type="application/json" (Gemini structured output)
  to reduce parsing failures.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from src.ingestion.pdf_loader import ParsedDocument

# Fixed schema of fields we want extracted for every industrial product.
# Kept intentionally generic across bearings/motors/sensors/etc so the same
# pipeline scales across a catalog rather than being product-type-specific.
PRODUCT_SCHEMA_FIELDS = [
    "product_name",
    "model_number",
    "manufacturer",
    "category",              # e.g. "bearing", "motor", "sensor"
    "short_description",
    "key_specifications",    # list of {parameter, value}
    "compliance_standards",  # list of strings, e.g. ["RoHS", "CE", "ISO 15:2017"]
    "operating_temperature_range",
    "protection_rating",     # e.g. IP55, IP67 if applicable
    "weight",
    "dimensions",
]

SYSTEM_INSTRUCTION = """You are a precise industrial product-data extraction engine.

You will be given raw text and table content extracted from a manufacturer's
technical datasheet PDF. Each chunk is labeled with a chunk_id and page number.

Your job: extract structured product intelligence as JSON matching the schema
described in the user prompt. For EVERY field you extract, you must cite the
exact chunk_id and page_number it came from.

Rules:
1. NEVER invent or infer a value that is not explicitly present in the source
   text. If a field is missing, blank, "N/A", or ambiguous in the source,
   return value=null, confidence=0.0, and leave source fields null too.
2. If a value is present but the source text itself is ambiguous or
   contradictory (e.g. two different units for what looks like the same
   quantity), still extract it, but set confidence <= 0.5 and add a short
   note in "flag" explaining the ambiguity.
3. confidence should be 1.0 only when the value is stated clearly, unambiguously,
   and in a single consistent unit/format.
4. Output STRICT JSON only. No markdown fences, no commentary outside the JSON.
5. For key_specifications, extract ALL parameter/value pairs you can find in
   spec tables, not just the ones named in the top-level schema fields.
"""

EXTRACTION_PROMPT_TEMPLATE = """Extract structured product intelligence from the
following document chunks (document: {doc_name}).

Return JSON with this exact shape:

{{
  "product_name": {{"value": str|null, "source_chunk_id": str|null, "source_page": int|null, "confidence": float, "flag": str|null}},
  "model_number": {{ ... same shape ... }},
  "manufacturer": {{ ... }},
  "category": {{ ... }},
  "short_description": {{ ... }},
  "operating_temperature_range": {{ ... }},
  "protection_rating": {{ ... }},
  "weight": {{ ... }},
  "dimensions": {{ ... }},
  "compliance_standards": {{"value": [str, ...]|null, "source_chunk_id": str|null, "source_page": int|null, "confidence": float, "flag": str|null}},
  "key_specifications": [
    {{"parameter": str, "value": str, "source_chunk_id": str, "source_page": int, "confidence": float, "flag": str|null}}
  ]
}}

--- DOCUMENT CHUNKS ---
{chunks_block}
--- END DOCUMENT CHUNKS ---
"""


def _format_chunks_block(doc: ParsedDocument) -> str:
    parts = []
    for c in doc.chunks:
        parts.append(
            f"[chunk_id={c.chunk_id} | page={c.page_number} | kind={c.kind}]\n{c.text}"
        )
    return "\n\n".join(parts)


@dataclass
class ExtractionResult:
    doc_id: str
    doc_name: str
    raw_json: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "fields": self.raw_json,
        }


class GeminiExtractor:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Put it in .env or pass api_key explicitly."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def extract(self, doc: ParsedDocument) -> ExtractionResult:
        chunks_block = _format_chunks_block(doc)
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            doc_name=doc.doc_name, chunks_block=chunks_block
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        text = response.text
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Gemini did not return valid JSON for {doc.doc_name}. "
                f"Raw response (first 500 chars): {text[:500]!r}"
            ) from e

        return ExtractionResult(doc_id=doc.doc_id, doc_name=doc.doc_name, raw_json=parsed)

    def extract_batch(self, docs: List[ParsedDocument]) -> List[ExtractionResult]:
        results = []
        for doc in docs:
            try:
                results.append(self.extract(doc))
            except Exception as e:
                print(f"[extractor] FAILED on {doc.doc_name}: {e}")
        return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Load .env manually (avoid adding python-dotenv as a hard dependency)
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    from src.ingestion.pdf_loader import load_pdfs_from_dir

    samples_dir = Path(__file__).resolve().parents[2] / "data" / "samples"
    docs = load_pdfs_from_dir(str(samples_dir))

    extractor = GeminiExtractor()
    results = extractor.extract_batch(docs)

    out_dir = Path(__file__).resolve().parents[2] / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        out_path = out_dir / f"{r.doc_id}_{Path(r.doc_name).stem}.json"
        out_path.write_text(json.dumps(r.to_dict(), indent=2))
        print(f"Wrote {out_path}")
