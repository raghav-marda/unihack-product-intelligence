# UniHack — AI-Powered Product Intelligence for Industrial Manufacturers

Built for the H2S UniHack hackathon (Aug 2026) by **Team ZeroTrace** — Raghav Marda, Sanjula, Hrishita, Himanshu.

## The problem

Industrial manufacturers sit on huge amounts of product data spread across
PDFs, catalogs, and spec sheets — mostly unstructured, inconsistent, and
painful to turn into clean, commerce-ready product records. Doing this by
hand doesn't scale past a handful of SKUs.

## What this does

Upload a product datasheet PDF, and the pipeline:

1. **Ingests** it — pulls out text and tables page by page, keeping track of
   exactly which page and chunk every piece of content came from.
2. **Extracts** structured product intelligence using Gemini — product name,
   model number, specs, compliance standards, etc — with every field tagged
   to its source page and a confidence score.
3. **Validates** the extraction with deterministic rules (no LLM involved
   here): flags missing fields, low-confidence extractions, and — the part
   we're most proud of — catches **unit inconsistencies** automatically
   (e.g. a datasheet quoting "18,000 rpm" in one place and "20000 min-1" in
   another for what looks like the same kind of measurement).
4. **Presents it for review** in a Streamlit dashboard — every field shown
   next to its source snippet, so nothing is a black box. Approve, edit, or
   export.

This isn't "call an LLM and hope." The validation layer is what makes the
output trustworthy enough to actually use downstream.

## Architecture

```
PDF datasheet
     │
     ▼
┌─────────────┐    page-tagged chunks (text + tables)
│  ingestion   │───────────────────────────┐
└─────────────┘                            │
                                            ▼
                                   ┌─────────────┐
                                   │  extraction  │  Gemini → structured JSON
                                   │  (Gemini)    │  + source citation + confidence
                                   └─────────────┘
                                            │
                                            ▼
                                   ┌─────────────┐
                                   │  validation  │  rule-based checks:
                                   │              │  missing / low-confidence /
                                   │              │  unit inconsistency / contradiction
                                   └─────────────┘
                                            │
                                            ▼
                                   ┌─────────────┐
                                   │  Streamlit   │  review, edit, approve, export
                                   │  dashboard   │
                                   └─────────────┘
```

## Project structure

```
src/
  ingestion/   pdf_loader.py     — PDF → page-tagged text/table chunks
  extraction/  extractor.py      — Gemini extraction agent, structured schema
  validation/  validator.py      — deterministic validation & consistency checks
  ui/          app.py            — Streamlit review dashboard
  pipeline.py                    — orchestrates ingest → extract → validate
scripts/
  generate_sample_data.py        — generates synthetic sample datasheets for testing
data/
  samples/                       — bundled sample PDFs (bearing, motor, sensor)
  output/                        — extraction results land here (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```
GEMINI_API_KEY=your_key_here
```

Get a free key from [Google AI Studio](https://aistudio.google.com/apikey).

## Running it

**CLI — process all bundled samples:**
```bash
python3 src/pipeline.py data/samples
```

**CLI — process a single PDF:**
```bash
python3 src/pipeline.py path/to/your/datasheet.pdf
```

**Dashboard:**
```bash
streamlit run src/ui/app.py
```
Upload a PDF (or click a bundled sample in the sidebar) and it runs the full
pipeline, then shows every field for review — value, source page, confidence,
and any validation flags.

## Sample data

We didn't have access to real manufacturer datasheets in time, so
`scripts/generate_sample_data.py` generates three realistic synthetic
datasheets (a bearing, a motor, and a proximity sensor) formatted like real
industrial spec sheets — including intentionally messy bits (a blank
efficiency-class field, mixed rpm/min-1 notation, inconsistent temperature
formatting) to actually exercise the validation layer instead of just
demoing on a clean happy path.

## What we'd add with more time

- Vector search across the whole catalog (chunking + embeddings) for
  cross-document queries, not just single-document extraction
- A proper ontology/schema per product category instead of one generic schema
- Batch upload + async processing for larger catalogs
- Persistent storage (currently JSON files on disk) for a real multi-user
  review workflow
