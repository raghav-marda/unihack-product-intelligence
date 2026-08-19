"""
Streamlit review dashboard for UniHack product intelligence pipeline.

Flow:
  1. Upload a product datasheet PDF (or pick from bundled samples).
  2. Run pipeline (ingest -> extract -> validate).
  3. Review extracted fields: value, source page/snippet, confidence,
     validation flags -- side by side, so nothing is a black box.
  4. Approve or manually edit any field.
  5. Export approved catalog record as JSON/CSV. Batch view shows all
     processed products together (the "scale across a catalog" story).

Run with:  streamlit run src/ui/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src.*` imports work when Streamlit runs this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.pdf_loader import load_pdf, ParsedDocument
from src.extraction.extractor import GeminiExtractor
from src.pipeline import process_document, load_env_file, OUTPUT_DIR

SAMPLES_DIR = REPO_ROOT / "data" / "samples"
UPLOAD_DIR = REPO_ROOT / "data" / "_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="UniHack Product Intelligence", layout="wide")


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
if "records" not in st.session_state:
    st.session_state.records = {}   # doc_name -> record dict (mutable, holds edits)
if "chunk_lookup" not in st.session_state:
    st.session_state.chunk_lookup = {}  # doc_name -> {chunk_id: text}


def get_extractor() -> GeminiExtractor | None:
    load_env_file()
    try:
        return GeminiExtractor()
    except ValueError as e:
        st.error(str(e))
        return None


def run_on_doc(doc: ParsedDocument) -> None:
    extractor = get_extractor()
    if extractor is None:
        return
    with st.spinner(f"Extracting {doc.doc_name}..."):
        try:
            record = process_document(doc, extractor)
        except Exception as e:
            st.error(f"Extraction failed for {doc.doc_name}: {e}")
            return
        record.save()
        st.session_state.records[doc.doc_name] = record.to_dict()
        st.session_state.chunk_lookup[doc.doc_name] = {
            c.chunk_id: c.text for c in doc.chunks
        }
    st.success(f"Done: {doc.doc_name}")


def load_existing_outputs() -> None:
    """Pre-load any already-processed JSON records from data/output, so the
    UI has something to show even before running a fresh extraction (also
    protects the demo if a live API call fails on stage)."""
    if not OUTPUT_DIR.exists():
        return
    for f in OUTPUT_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            doc_name = data.get("doc_name")
            if doc_name and doc_name not in st.session_state.records:
                st.session_state.records[doc_name] = data
        except Exception:
            continue


load_existing_outputs()


# ---------------------------------------------------------------------------
# Sidebar: ingest controls
# ---------------------------------------------------------------------------
st.sidebar.title("UniHack — Product Intelligence")
st.sidebar.caption("AI-powered extraction for industrial product datasheets")

st.sidebar.subheader("1. Add a document")

uploaded = st.sidebar.file_uploader("Upload a datasheet PDF", type=["pdf"])
if uploaded is not None:
    dest = UPLOAD_DIR / uploaded.name
    dest.write_bytes(uploaded.getvalue())
    if st.sidebar.button(f"Run extraction on '{uploaded.name}'"):
        doc = load_pdf(str(dest))
        run_on_doc(doc)

st.sidebar.markdown("---")
st.sidebar.caption("Or use a bundled sample:")
sample_files = sorted(SAMPLES_DIR.glob("*.pdf")) if SAMPLES_DIR.exists() else []
for sample_path in sample_files:
    if st.sidebar.button(f"Run: {sample_path.name}"):
        doc = load_pdf(str(sample_path))
        run_on_doc(doc)

if st.sidebar.button("Run ALL bundled samples (batch demo)"):
    for sample_path in sample_files:
        doc = load_pdf(str(sample_path))
        run_on_doc(doc)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Product Intelligence Review")

if not st.session_state.records:
    st.info(
        "No products processed yet. Upload a datasheet or run a bundled "
        "sample from the sidebar to get started."
    )
    st.stop()

doc_names = list(st.session_state.records.keys())
tab_labels = ["📋 Catalog Overview"] + doc_names
tabs = st.tabs(tab_labels)

# --- Catalog overview tab ---
with tabs[0]:
    st.subheader("All processed products")
    rows = []
    for name, rec in st.session_state.records.items():
        val = rec.get("validation", {})
        fields = rec.get("extraction", {})
        rows.append({
            "Document": name,
            "Product": (fields.get("product_name") or {}).get("value") or "—",
            "Model": (fields.get("model_number") or {}).get("value") or "—",
            "Completeness": f"{val.get('completeness_score', 0) * 100:.0f}%",
            "Issues": val.get("issue_count", 0),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)

    # Export full catalog
    all_export = [
        {"doc_name": name, **rec}
        for name, rec in st.session_state.records.items()
    ]
    st.download_button(
        "⬇️ Export full catalog (JSON)",
        data=json.dumps(all_export, indent=2),
        file_name="unihack_catalog_export.json",
        mime="application/json",
    )

# --- Per-document review tabs ---
FIELD_LABELS = {
    "product_name": "Product Name",
    "model_number": "Model Number",
    "manufacturer": "Manufacturer",
    "category": "Category",
    "short_description": "Description",
    "operating_temperature_range": "Operating Temp. Range",
    "protection_rating": "Protection Rating",
    "weight": "Weight",
    "dimensions": "Dimensions",
    "compliance_standards": "Compliance Standards",
}

for tab, doc_name in zip(tabs[1:], doc_names):
    with tab:
        rec = st.session_state.records[doc_name]
        fields = rec.get("extraction", {})
        validation = rec.get("validation", {})
        chunk_lookup = st.session_state.chunk_lookup.get(doc_name, {})

        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.subheader(doc_name)
        with col_b:
            score = validation.get("completeness_score", 0) * 100
            st.metric("Completeness", f"{score:.0f}%")

        issues = validation.get("issues", [])
        if issues:
            with st.expander(f"⚠️ {len(issues)} validation issue(s) found", expanded=True):
                for issue in issues:
                    icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(
                        issue["severity"], "⚪"
                    )
                    st.markdown(f"{icon} **{issue['field']}** ({issue['issue_type']}): {issue['message']}")
        else:
            st.success("No validation issues found.")

        st.markdown("### Extracted Fields")
        st.caption("Click a field's source page to see the exact text it was extracted from.")

        for fkey, flabel in FIELD_LABELS.items():
            entry = fields.get(fkey) or {}
            value = entry.get("value")
            confidence = entry.get("confidence", 0.0)
            source_page = entry.get("source_page")
            source_chunk_id = entry.get("source_chunk_id")
            flag = entry.get("flag")

            is_list_field = isinstance(value, list)
            if is_list_field:
                display_value = ", ".join(value) if value else None
            else:
                display_value = value

            c1, c2, c3 = st.columns([2, 3, 2])
            with c1:
                st.markdown(f"**{flabel}**")
            with c2:
                if display_value:
                    edited = st.text_input(
                        flabel, value=str(display_value),
                        key=f"{doc_name}::{fkey}", label_visibility="collapsed",
                    )
                    if is_list_field:
                        # Preserve the original list type — writing the raw
                        # comma-joined string back here would silently
                        # corrupt the field's schema (list -> str) every
                        # time this widget reruns, which happens on every
                        # interaction with the page, not just on an
                        # intentional edit to this specific field.
                        fields[fkey]["value"] = [
                            v.strip() for v in edited.split(",") if v.strip()
                        ]
                    else:
                        fields[fkey]["value"] = edited
                else:
                    st.markdown("_Not found in document_")
            with c3:
                conf_color = "🟢" if confidence >= 0.8 else ("🟡" if confidence >= 0.5 else "🔴")
                st.markdown(f"{conf_color} conf {confidence:.2f} · p.{source_page or '—'}")

            if flag:
                st.caption(f"⚑ {flag}")
            if source_chunk_id and source_chunk_id in chunk_lookup:
                with st.expander("View source snippet", expanded=False):
                    st.text(chunk_lookup[source_chunk_id][:500])

            st.markdown("---")

        st.markdown("### Key Specifications")
        specs = fields.get("key_specifications", []) or []
        if specs:
            spec_df = pd.DataFrame([
                {
                    "Parameter": s.get("parameter"),
                    "Value": s.get("value"),
                    "Page": s.get("source_page"),
                    "Confidence": s.get("confidence"),
                    "Flag": s.get("flag") or "",
                }
                for s in specs
            ])
            st.dataframe(spec_df, width='stretch', hide_index=True)
        else:
            st.caption("No key specifications extracted.")

        colx, coly = st.columns(2)
        with colx:
            if st.button("✅ Approve record", key=f"approve_{doc_name}"):
                out_path = OUTPUT_DIR / f"approved_{Path(doc_name).stem}.json"
                out_path.write_text(json.dumps(rec, indent=2))
                st.success(f"Approved and saved to {out_path.name}")
        with coly:
            st.download_button(
                "⬇️ Export this record (JSON)",
                data=json.dumps(rec, indent=2),
                file_name=f"{Path(doc_name).stem}_export.json",
                mime="application/json",
                key=f"export_{doc_name}",
            )
