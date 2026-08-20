"""
Catalog search — RAG over the processed product catalog.

Everything built so far operates on ONE document at a time: ingest one PDF,
extract one product record, validate it, review it. That's necessary but
not sufficient against the actual problem statement, which explicitly
calls out "AI agents, RAG, knowledge graphs" as expected solution
approaches and asks the system to "scale efficiently across large product
catalogs." A pipeline that only ever looks at one document in isolation
doesn't really answer that — a catalog manager's real question is usually
cross-document: "which of our products are rated IP65 or higher?", "show
me every bearing with a dynamic load rating above 10kN."

This module adds that layer: every approved product record gets embedded
(via Gemini's embedding model) and stored in a local vector index
(ChromaDB). Natural-language queries get embedded the same way and matched
against the index, so search isn't limited to exact keyword matches — the
embeddings actually understand semantic similarity between products.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from google.genai.errors import APIError

EMBEDDING_MODEL = "gemini-embedding-001"


def compose_product_text(record: Dict[str, Any]) -> str:
    """Build a single text blob representing a product record, used both
    for indexing (RETRIEVAL_DOCUMENT) and implicitly matched against at
    query time. Keeping this in one function means indexing and search
    always agree on what a "product" looks like as text."""
    fields = record.get("extraction", {})

    def _val(fname: str) -> str:
        entry = fields.get(fname)
        if isinstance(entry, dict) and entry.get("value"):
            return str(entry["value"])
        return ""

    parts = [
        _val("product_name"),
        _val("manufacturer"),
        f"Category: {_val('category')}" if _val("category") else "",
        _val("short_description"),
        f"Model: {_val('model_number')}" if _val("model_number") else "",
        f"Protection rating: {_val('protection_rating')}" if _val("protection_rating") else "",
        f"Operating temperature: {_val('operating_temperature_range')}"
        if _val("operating_temperature_range") else "",
    ]

    compliance = fields.get("compliance_standards", {})
    if isinstance(compliance, dict) and compliance.get("value"):
        parts.append("Compliance: " + ", ".join(compliance["value"]))

    for spec in fields.get("key_specifications", []) or []:
        if isinstance(spec, dict) and spec.get("parameter") and spec.get("value"):
            parts.append(f"{spec['parameter']}: {spec['value']}")

    return "\n".join(p for p in parts if p)


@dataclass
class SearchResult:
    doc_name: str
    product_name: Optional[str]
    category: Optional[str]
    model_number: Optional[str]
    distance: float
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_name": self.doc_name,
            "product_name": self.product_name,
            "category": self.category,
            "model_number": self.model_number,
            "distance": round(self.distance, 4),
            "snippet": self.snippet,
        }


class CatalogIndex:
    """Wraps a Chroma collection + Gemini embedding client. Uses an
    in-memory (ephemeral) Chroma client by default so it's trivially
    testable without touching disk; pass persist_dir for a real run where
    the index should survive across sessions."""

    def __init__(self, api_key: Optional[str] = None, collection_name: str = "spectrace_catalog",
                 persist_dir: Optional[str] = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Put it in .env or pass api_key explicitly."
            )
        self.genai_client = genai.Client(api_key=api_key)

        if persist_dir:
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.chroma_client = chromadb.EphemeralClient()

        self.collection = self.chroma_client.get_or_create_collection(collection_name)

    @retry(
        retry=retry_if_exception_type((APIError, TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _embed_texts(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Isolated so tests can mock exactly this call without needing to
        fake the whole genai client."""
        response = self.genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]

    def add_product(self, record: Dict[str, Any]) -> None:
        self.add_products([record])

    def add_products(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        texts = [compose_product_text(r) for r in records]
        embeddings = self._embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

        ids, metadatas, documents = [], [], []
        for record, text in zip(records, texts):
            doc_id = record.get("doc_id") or record.get("doc_name")
            fields = record.get("extraction", {})
            product_name = fields.get("product_name", {}).get("value") if isinstance(
                fields.get("product_name"), dict) else None
            category = fields.get("category", {}).get("value") if isinstance(
                fields.get("category"), dict) else None
            model_number = fields.get("model_number", {}).get("value") if isinstance(
                fields.get("model_number"), dict) else None

            ids.append(doc_id)
            documents.append(text)
            metadatas.append({
                "doc_name": record.get("doc_name", ""),
                "product_name": product_name or "",
                "category": category or "",
                "model_number": model_number or "",
            })

        self.collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas,
        )

    def search(self, query: str, n_results: int = 5) -> List[SearchResult]:
        if self.collection.count() == 0:
            return []
        n_results = min(n_results, self.collection.count())

        query_embedding = self._embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=n_results,
        )

        output = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for doc_id, distance, document, metadata in zip(ids, distances, documents, metadatas):
            output.append(SearchResult(
                doc_name=metadata.get("doc_name", doc_id),
                product_name=metadata.get("product_name") or None,
                category=metadata.get("category") or None,
                model_number=metadata.get("model_number") or None,
                distance=distance,
                snippet=document[:300],
            ))
        return output

    def count(self) -> int:
        return self.collection.count()
