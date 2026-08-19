"""
PDF ingestion module.

Extracts text content page-by-page (and tables where detectable) from a PDF,
producing a list of "chunks" — each chunk carries the source page number and
a stable chunk_id. This page/chunk-level metadata is what lets the extraction
agent later cite exactly where a field came from (traceability requirement).

Uses PyMuPDF (fitz) for text extraction because it's fast and preserves
reading order reasonably well for structured datasheets. Falls back to
pdfplumber for table-heavy pages if PyMuPDF text looks too sparse.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import List, Optional

import pymupdf  # PyMuPDF (modern import name; avoids deprecated `fitz` alias)
import pdfplumber


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int          # 1-indexed
    text: str
    kind: str = "text"        # "text" | "table"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "text": self.text,
            "kind": self.kind,
        }


@dataclass
class ParsedDocument:
    doc_id: str
    doc_name: str
    file_path: str
    num_pages: int
    chunks: List[Chunk] = field(default_factory=list)

    def full_text(self) -> str:
        return "\n\n".join(c.text for c in self.chunks)


def _make_doc_id(file_path: str) -> str:
    """Stable short id derived from file path + size, so re-ingesting the
    same file doesn't create duplicate ids across runs."""
    stat = os.stat(file_path)
    key = f"{os.path.basename(file_path)}::{stat.st_size}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def _table_to_text(table: List[List[Optional[str]]]) -> str:
    """Render an extracted table as a simple pipe-delimited text block so it
    stays LLM-friendly while remaining traceable to its source page."""
    lines = []
    for row in table:
        cells = [(c or "").strip() for c in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def load_pdf(file_path: str, min_chars_for_text_page: int = 40) -> ParsedDocument:
    """
    Parse a PDF into a ParsedDocument of page-scoped chunks.

    For each page:
      1. Extract plain text via PyMuPDF.
      2. Also attempt table extraction via pdfplumber; if tables are found,
         add them as separate "table" chunks (in addition to page text),
         since tabular spec data is often the most extraction-critical part.

    NOTE ON INTENTIONAL OVERLAP: a page's raw text chunk and its table
    chunk(s) will contain overlapping content — a spec value inside a
    table shows up both in the messy raw-text rendering of the page AND
    in the cleaned, row-by-row table chunk. This is deliberate, not an
    oversight: raw PDF text extraction frequently mangles table structure
    (columns collapse, spacing gets lost), so pdfplumber's cleaner
    row-by-row rendering exists specifically to give the extraction agent
    a reliable structured alternative for exact values. The overlap does
    cost some extra tokens per call, but a naive de-duplication pass here
    (regex/line-diffing the two versions against each other) is fragile
    enough — different whitespace, cell ordering, wrapped cells — that it
    risks silently deleting legitimate prose alongside table rows. Instead,
    the extraction agent's system instruction (see extractor.py)
    explicitly tells the model to prefer citing the "table" chunk_id over
    the "text" chunk_id when a value is available in both, which resolves
    the ambiguity at the point that actually matters (citation accuracy)
    without risking data loss at ingestion time.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc_id = _make_doc_id(file_path)
    doc_name = os.path.basename(file_path)

    chunks: List[Chunk] = []

    fitz_doc = pymupdf.open(file_path)
    num_pages = fitz_doc.page_count

    with pdfplumber.open(file_path) as plumber_doc:
        for page_index in range(num_pages):
            page_number = page_index + 1

            # --- plain text (PyMuPDF) ---
            fitz_page = fitz_doc.load_page(page_index)
            text = fitz_page.get_text("text").strip()

            if len(text) >= min_chars_for_text_page:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}-p{page_number}-text",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page_number=page_number,
                    text=text,
                    kind="text",
                ))

            # --- tables (pdfplumber) ---
            try:
                plumber_page = plumber_doc.pages[page_index]
                tables = plumber_page.extract_tables()
            except Exception:
                tables = []

            for t_idx, table in enumerate(tables):
                table_text = _table_to_text(table)
                if table_text.strip():
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}-p{page_number}-table{t_idx}",
                        doc_id=doc_id,
                        doc_name=doc_name,
                        page_number=page_number,
                        text=table_text,
                        kind="table",
                    ))

    fitz_doc.close()

    return ParsedDocument(
        doc_id=doc_id,
        doc_name=doc_name,
        file_path=file_path,
        num_pages=num_pages,
        chunks=chunks,
    )


def load_pdfs_from_dir(dir_path: str) -> List[ParsedDocument]:
    """Convenience batch loader for a directory of PDFs (used for the
    'scale across multiple products' demo requirement)."""
    docs = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.lower().endswith(".pdf"):
            docs.append(load_pdf(os.path.join(dir_path, fname)))
    return docs


if __name__ == "__main__":
    # Quick self-test against the generated sample datasheets.
    samples_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "samples"
    )
    samples_dir = os.path.abspath(samples_dir)
    docs = load_pdfs_from_dir(samples_dir)
    for d in docs:
        print(f"\n=== {d.doc_name} ({d.num_pages} pages, {len(d.chunks)} chunks) ===")
        for c in d.chunks[:3]:
            preview = c.text[:80].replace("\n", " ")
            print(f"  [{c.kind}] page {c.page_number} chunk_id={c.chunk_id} :: {preview}...")
