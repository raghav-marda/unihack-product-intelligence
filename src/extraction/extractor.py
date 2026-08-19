"""
Extraction agent (v2 — hardened).

Takes a ParsedDocument (from pdf_loader) and calls Gemini to produce a
structured product-intelligence record.

What changed from v1, and why:

1. STRICT SCHEMA VALIDATION (Pydantic + response_schema).
   v1 just did json.loads() on whatever Gemini returned and trusted the
   shape completely. If the model ever drifted from the expected shape
   (missing key, wrong type, confidence as a string instead of a float),
   that bad data would flow silently into validation.py and the UI,
   surfacing as a confusing downstream crash or — worse — silently wrong
   output. Now the response is validated against a Pydantic model
   (ProductExtraction) immediately, AND we pass that same schema to Gemini
   via response_schema so the model is constrained on the way out too.
   A malformed response now fails loudly, at the extraction boundary,
   with a clear error.

2. RETRY WITH BACKOFF.
   v1 made a single API call with no retry. Transient failures (rate
   limits, momentary network issues) would kill the whole batch. Now
   wrapped with tenacity: retries transient/network errors up to 3 times
   with exponential backoff. Schema validation failures are NOT retried
   (retrying won't fix a systematically malformed response).

3. CHUNK BATCHING FOR LARGE DOCUMENTS.
   v1 stuffed every chunk from a document into a single prompt,
   regardless of size. For a large multi-page catalog entry (or a
   scanned document with lots of OCR text) this risks degraded
   extraction quality well before hitting a hard context limit. Now
   chunks are batched by a character budget; each batch is extracted
   independently and results are merged, keeping the highest-confidence
   value per field across batches.

4. CITATION PREFERENCE.
   pdf_loader.py deliberately extracts both raw page text AND a
   separately-cleaned version of any tables on that page (see the note
   in pdf_loader.py). That means the same spec value can appear in two
   chunks: a messy "text" chunk and a clean "table" chunk. The system
   instruction now explicitly tells the model to cite the "table" chunk
   when a value is available in both, since it's the more reliable
   source for exact values.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.ingestion.pdf_loader import Chunk, ParsedDocument

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class FieldExtraction(BaseModel):
    value: Optional[str] = None
    source_chunk_id: Optional[str] = None
    source_page: Optional[int] = None
    confidence: float = 0.0
    flag: Optional[str] = None


class ComplianceExtraction(BaseModel):
    value: Optional[List[str]] = None
    source_chunk_id: Optional[str] = None
    source_page: Optional[int] = None
    confidence: float = 0.0
    flag: Optional[str] = None


class KeySpecExtraction(BaseModel):
    parameter: str
    value: str
    source_chunk_id: str
    source_page: int
    confidence: float = 0.0
    flag: Optional[str] = None


class ProductExtraction(BaseModel):
    product_name: FieldExtraction = Field(default_factory=FieldExtraction)
    model_number: FieldExtraction = Field(default_factory=FieldExtraction)
    manufacturer: FieldExtraction = Field(default_factory=FieldExtraction)
    category: FieldExtraction = Field(default_factory=FieldExtraction)
    short_description: FieldExtraction = Field(default_factory=FieldExtraction)
    operating_temperature_range: FieldExtraction = Field(default_factory=FieldExtraction)
    protection_rating: FieldExtraction = Field(default_factory=FieldExtraction)
    weight: FieldExtraction = Field(default_factory=FieldExtraction)
    dimensions: FieldExtraction = Field(default_factory=FieldExtraction)
    compliance_standards: ComplianceExtraction = Field(default_factory=ComplianceExtraction)
    key_specifications: List[KeySpecExtraction] = Field(default_factory=list)


TOP_LEVEL_SINGLE_FIELDS = [
    "product_name", "model_number", "manufacturer", "category",
    "short_description", "operating_temperature_range", "protection_rating",
    "weight", "dimensions",
]

# Character budget per extraction call. Conservative on purpose: staying
# well under the model's real context limit keeps extraction quality high
# rather than testing how close to the ceiling we can get.
MAX_CHARS_PER_BATCH = 12_000

SYSTEM_INSTRUCTION = """You are a precise industrial product-data extraction engine.

You will be given raw text and table content extracted from a manufacturer's
technical datasheet PDF. Each chunk is labeled with a chunk_id, page number,
and a "kind" of either "text" (raw page text) or "table" (a cleaned,
row-by-row rendering of a detected table on that page).

Note: the same value may appear in BOTH a "text" chunk and a "table" chunk
for the same page, because the table chunk is a cleaner re-rendering of
structured data that also appears in the page's raw text. When a value is
available in both, ALWAYS cite the "table" chunk_id — it is the more
reliable source for exact numeric/spec values.

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
4. Output STRICT JSON only, matching the provided schema exactly. No markdown
   fences, no commentary outside the JSON.
5. For key_specifications, extract ALL parameter/value pairs you can find in
   spec tables, not just the ones named in the top-level schema fields.
6. If a chunk_id you would cite is not present in the input, do not cite it —
   leave source_chunk_id null instead of guessing an id.
"""

EXTRACTION_PROMPT_TEMPLATE = """Extract structured product intelligence from the
following document chunks (document: {doc_name}{batch_note}).

Return JSON matching the ProductExtraction schema exactly.

--- DOCUMENT CHUNKS ---
{chunks_block}
--- END DOCUMENT CHUNKS ---
"""


def _format_chunk(c: Chunk) -> str:
    return f"[chunk_id={c.chunk_id} | page={c.page_number} | kind={c.kind}]\n{c.text}"


def _format_chunks_block(chunks: List[Chunk]) -> str:
    return "\n\n".join(_format_chunk(c) for c in chunks)


def _batch_chunks(chunks: List[Chunk], max_chars: int = MAX_CHARS_PER_BATCH) -> List[List[Chunk]]:
    """Greedily group chunks into batches that stay under a character
    budget. A single oversized chunk still gets its own batch rather than
    being dropped or truncated."""
    batches: List[List[Chunk]] = []
    current: List[Chunk] = []
    current_len = 0

    for chunk in chunks:
        chunk_len = len(chunk.text)
        if current and current_len + chunk_len > max_chars:
            batches.append(current)
            current = []
            current_len = 0
        current.append(chunk)
        current_len += chunk_len

    if current:
        batches.append(current)

    return batches or [[]]


def _merge_field(a: FieldExtraction, b: FieldExtraction) -> FieldExtraction:
    """Keep whichever field has a non-null value and higher confidence."""
    if a.value is None and b.value is None:
        return a if a.confidence >= b.confidence else b
    if a.value is None:
        return b
    if b.value is None:
        return a
    return a if a.confidence >= b.confidence else b


def _merge_extractions(results: List[ProductExtraction]) -> ProductExtraction:
    """Merge per-batch extraction results into a single record. For scalar
    fields, keep the higher-confidence non-null value. For compliance
    standards, take the union. For key_specifications, concatenate and
    de-duplicate by normalized parameter name, keeping the higher-confidence
    entry when the same parameter appears in multiple batches."""
    if len(results) == 1:
        return results[0]

    merged_kwargs: Dict[str, Any] = {}
    for fname in TOP_LEVEL_SINGLE_FIELDS:
        best = results[0].__getattribute__(fname)
        for r in results[1:]:
            best = _merge_field(best, r.__getattribute__(fname))
        merged_kwargs[fname] = best

    # compliance_standards: union of values, keep best confidence/source
    compliance_values: List[str] = []
    best_compliance = results[0].compliance_standards
    for r in results:
        c = r.compliance_standards
        if c.value:
            for v in c.value:
                if v not in compliance_values:
                    compliance_values.append(v)
        if c.confidence > best_compliance.confidence:
            best_compliance = c
    merged_compliance = ComplianceExtraction(
        value=compliance_values or None,
        source_chunk_id=best_compliance.source_chunk_id,
        source_page=best_compliance.source_page,
        confidence=best_compliance.confidence,
        flag=best_compliance.flag,
    )
    merged_kwargs["compliance_standards"] = merged_compliance

    # key_specifications: de-dupe by normalized parameter name
    specs_by_param: Dict[str, KeySpecExtraction] = {}
    for r in results:
        for spec in r.key_specifications:
            key = spec.parameter.strip().lower()
            existing = specs_by_param.get(key)
            if existing is None or spec.confidence > existing.confidence:
                specs_by_param[key] = spec
    merged_kwargs["key_specifications"] = list(specs_by_param.values())

    return ProductExtraction(**merged_kwargs)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class ExtractionResult:
    def __init__(self, doc_id: str, doc_name: str, extraction: ProductExtraction):
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.extraction = extraction

    @property
    def raw_json(self) -> Dict[str, Any]:
        """Dict view matching the same shape v1 produced, so pipeline.py,
        validator.py, and the Streamlit UI don't need to change."""
        return self.extraction.model_dump()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "fields": self.raw_json,
        }


def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient API/network errors. Do NOT retry on schema
    validation failures — those are deterministic and retrying just burns
    quota for the same failure."""
    if isinstance(exc, ValidationError):
        return False
    if isinstance(exc, APIError):
        # 429 (rate limit) and 5xx (transient server errors) are worth
        # retrying; 4xx client errors (bad request, bad key) are not.
        status = getattr(exc, "code", None)
        if status is not None:
            return status == 429 or status >= 500
        return True
    return isinstance(exc, (TimeoutError, ConnectionError))


class GeminiExtractor:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Put it in .env or pass api_key explicitly."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    @retry(
        retry=retry_if_exception_type((APIError, TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_gemini(self, prompt: str) -> types.GenerateContentResponse:
        return self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=ProductExtraction,
                temperature=0.1,
            ),
        )

    def _extract_batch_of_chunks(self, doc_name: str, chunks: List[Chunk],
                                  batch_note: str = "") -> ProductExtraction:
        chunks_block = _format_chunks_block(chunks)
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            doc_name=doc_name, batch_note=batch_note, chunks_block=chunks_block
        )

        response = self._call_gemini(prompt)

        # Prefer the SDK's own parsed pydantic instance when available —
        # it validates against response_schema for us. Fall back to a
        # manual parse + validate so we still get a clear error either way.
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, ProductExtraction):
            return parsed

        text = response.text
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Gemini did not return valid JSON for {doc_name}. "
                f"Raw response (first 500 chars): {text[:500]!r}"
            ) from e

        try:
            return ProductExtraction.model_validate(raw)
        except ValidationError as e:
            raise ValueError(
                f"Gemini's response for {doc_name} did not match the expected "
                f"schema: {e}"
            ) from e

    def extract(self, doc: ParsedDocument) -> ExtractionResult:
        batches = _batch_chunks(doc.chunks)

        if len(batches) == 1:
            result = self._extract_batch_of_chunks(doc.doc_name, batches[0])
        else:
            batch_results = []
            for i, batch in enumerate(batches, start=1):
                note = f", batch {i}/{len(batches)} of a large document"
                batch_results.append(
                    self._extract_batch_of_chunks(doc.doc_name, batch, batch_note=note)
                )
            result = _merge_extractions(batch_results)

        return ExtractionResult(doc_id=doc.doc_id, doc_name=doc.doc_name, extraction=result)

    def extract_batch(self, docs: List[ParsedDocument]) -> List[ExtractionResult]:
        results = []
        for doc in docs:
            try:
                results.append(self.extract(doc))
            except Exception as e:
                print(f"[extractor] FAILED on {doc.doc_name}: {e}")
        return results


if __name__ == "__main__":
    from src.env_utils import load_env_file
    from src.ingestion.pdf_loader import load_pdfs_from_dir

    load_env_file()

    repo_root = Path(__file__).resolve().parents[2]
    samples_dir = repo_root / "data" / "samples"
    docs = load_pdfs_from_dir(str(samples_dir))

    extractor = GeminiExtractor()
    results = extractor.extract_batch(docs)

    out_dir = repo_root / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        out_path = out_dir / f"{r.doc_id}_{Path(r.doc_name).stem}.json"
        out_path.write_text(json.dumps(r.to_dict(), indent=2))
        print(f"Wrote {out_path}")
