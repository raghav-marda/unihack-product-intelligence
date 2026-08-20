from unittest.mock import MagicMock, patch

import pytest

from src.search.catalog_index import CatalogIndex, compose_product_text


def _record(doc_id, doc_name, product_name, category, model_number,
            key_specs=None, compliance=None):
    return {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "extraction": {
            "product_name": {"value": product_name},
            "manufacturer": {"value": "Test Manufacturer"},
            "category": {"value": category},
            "model_number": {"value": model_number},
            "protection_rating": {"value": None},
            "operating_temperature_range": {"value": None},
            "compliance_standards": {"value": compliance or []},
            "key_specifications": key_specs or [],
        },
    }


BEARING = _record(
    "bearing1", "bearing.pdf", "Deep Groove Ball Bearing", "bearing", "SKB-6205-2RS",
    key_specs=[{"parameter": "Dynamic Load Rating", "value": "14000 N"}],
    compliance=["ISO 15:2017"],
)
SENSOR = _record(
    "sensor1", "sensor.pdf", "Inductive Proximity Sensor", "sensor", "IPS-18-M-DC-NO",
    key_specs=[{"parameter": "Operating Voltage", "value": "10-30 VDC"}],
    compliance=["CE"],
)


def _fake_embed_factory():
    """Deterministic fake embeddings: anything bearing-related gets one
    vector, anything sensor-related gets an orthogonal one, so cosine/L2
    distance in the test is unambiguous and doesn't depend on a real model."""
    def fake_embed(texts, task_type):
        out = []
        for t in texts:
            tl = t.lower()
            if "bearing" in tl or "load rating" in tl:
                out.append([1.0, 0.0, 0.0])
            elif "sensor" in tl or "proximity" in tl or "voltage" in tl:
                out.append([0.0, 1.0, 0.0])
            else:
                out.append([0.5, 0.5, 0.0])
        return out
    return fake_embed


@pytest.fixture
def index():
    with patch.object(CatalogIndex, "__init__", lambda self, **kw: None):
        idx = CatalogIndex.__new__(CatalogIndex)
        import chromadb
        idx.chroma_client = chromadb.EphemeralClient()
        idx.collection = idx.chroma_client.get_or_create_collection(
            f"test_{id(idx)}"  # unique per test to avoid cross-test collisions
        )
        idx.genai_client = MagicMock()
        idx._embed_texts = _fake_embed_factory()
        yield idx


# ---------------------------------------------------------------------------
# compose_product_text
# ---------------------------------------------------------------------------

def test_compose_product_text_includes_key_fields():
    text = compose_product_text(BEARING)
    assert "Deep Groove Ball Bearing" in text
    assert "bearing" in text.lower()
    assert "SKB-6205-2RS" in text
    assert "Dynamic Load Rating" in text
    assert "14000 N" in text
    assert "ISO 15:2017" in text


def test_compose_product_text_skips_missing_fields_without_crashing():
    minimal = {"doc_id": "x", "doc_name": "x.pdf", "extraction": {}}
    text = compose_product_text(minimal)  # should not raise
    assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def test_add_products_indexes_all_records(index):
    index.add_products([BEARING, SENSOR])
    assert index.count() == 2


def test_add_product_single_record(index):
    index.add_product(BEARING)
    assert index.count() == 1


def test_add_products_empty_list_is_a_noop(index):
    index.add_products([])
    assert index.count() == 0


def test_reindexing_same_doc_id_upserts_not_duplicates(index):
    """Re-running extraction on the same document (e.g. after an edit)
    should update its index entry, not create a duplicate."""
    index.add_products([BEARING])
    assert index.count() == 1
    updated_bearing = dict(BEARING)
    updated_bearing["extraction"] = dict(BEARING["extraction"])
    updated_bearing["extraction"]["product_name"] = {"value": "Updated Bearing Name"}
    index.add_products([updated_bearing])
    assert index.count() == 1  # still 1, not 2


# ---------------------------------------------------------------------------
# Search / ranking
# ---------------------------------------------------------------------------

def test_search_empty_index_returns_empty_list(index):
    results = index.search("anything")
    assert results == []


def test_search_ranks_semantically_relevant_product_first(index):
    index.add_products([BEARING, SENSOR])
    results = index.search("heavy duty bearing with high load rating", n_results=2)
    assert len(results) == 2
    assert results[0].product_name == "Deep Groove Ball Bearing"
    assert results[0].distance < results[1].distance


def test_search_result_includes_metadata(index):
    index.add_products([SENSOR])
    results = index.search("proximity sensor", n_results=1)
    assert results[0].category == "sensor"
    assert results[0].model_number == "IPS-18-M-DC-NO"
    assert results[0].doc_name == "sensor.pdf"


def test_search_n_results_capped_at_available_count(index):
    """Asking for more results than exist in the index shouldn't error."""
    index.add_products([BEARING])
    results = index.search("bearing", n_results=10)
    assert len(results) == 1


def test_missing_api_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            CatalogIndex()
