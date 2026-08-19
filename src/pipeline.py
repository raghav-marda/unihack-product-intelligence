"""
Pipeline orchestrator.

Runs the full flow for one or many PDFs:
    ingest (pdf_loader) -> extract (extractor, Gemini) -> validate (validator)
and writes a combined per-product JSON record to data/output/, which the
Streamlit review UI reads from.

This is the single entry point both the CLI and the UI call, so ingestion/
extraction/validation logic never has to be duplicated.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.ingestion.pdf_loader import ParsedDocument, load_pdf, load_pdfs_from_dir
from src.extraction.extractor import GeminiExtractor, ExtractionResult
from src.validation.validator import ValidationReport, validate_extraction

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "output"
ENV_PATH = REPO_ROOT / ".env"


def load_env_file(env_path: Path = ENV_PATH) -> None:
    """Minimal .env loader (no python-dotenv dependency needed)."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


@dataclass
class ProductRecord:
    doc_id: str
    doc_name: str
    extraction: dict
    validation: dict

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "extraction": self.extraction,
            "validation": self.validation,
        }

    def save(self, output_dir: Path = OUTPUT_DIR) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self.doc_name).stem
        out_path = output_dir / f"{self.doc_id}_{stem}.json"
        out_path.write_text(json.dumps(self.to_dict(), indent=2))
        return out_path


def process_document(doc: ParsedDocument, extractor: GeminiExtractor) -> ProductRecord:
    extraction: ExtractionResult = extractor.extract(doc)
    fields = extraction.raw_json
    validation: ValidationReport = validate_extraction(doc.doc_name, fields)

    return ProductRecord(
        doc_id=doc.doc_id,
        doc_name=doc.doc_name,
        extraction=fields,
        validation=validation.to_dict(),
    )


def run_pipeline_on_file(file_path: str, api_key: Optional[str] = None) -> ProductRecord:
    load_env_file()
    doc = load_pdf(file_path)
    extractor = GeminiExtractor(api_key=api_key)
    record = process_document(doc, extractor)
    record.save()
    return record


def run_pipeline_on_dir(dir_path: str, api_key: Optional[str] = None) -> List[ProductRecord]:
    load_env_file()
    docs = load_pdfs_from_dir(dir_path)
    extractor = GeminiExtractor(api_key=api_key)

    records = []
    for doc in docs:
        try:
            record = process_document(doc, extractor)
            record.save()
            records.append(record)
            print(f"[pipeline] OK: {doc.doc_name} "
                  f"(completeness={record.validation['completeness_score']}, "
                  f"issues={record.validation['issue_count']})")
        except Exception as e:
            print(f"[pipeline] FAILED: {doc.doc_name} -> {e}")
    return records


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "data" / "samples")
    if os.path.isdir(target):
        run_pipeline_on_dir(target)
    else:
        rec = run_pipeline_on_file(target)
        print(f"Saved: {rec.doc_name}")
