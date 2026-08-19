"""Unit tests for src.vectorstore.chroma_store — real Chroma, in-memory (no server)."""

import uuid

import pytest

from src.chunking.splitter import Chunk
from src.vectorstore.chroma_store import ChromaVectorStore


def make_chunk(text: str, chunk_id: str, page: int = 1, source: str = "book.pdf") -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, page_number=page, source_file=source)


def make_store() -> ChromaVectorStore:
    # Unique name per store: EphemeralClient instances in one process share an
    # in-memory DB, so a fixed collection name would leak state across tests.
    return ChromaVectorStore(collection_name=f"test_{uuid.uuid4().hex}", persist_dir=None)


def test_add_and_query_top1():
    store = make_store()
    store.add(
        [
            make_chunk("alpha", "a"),
            make_chunk("beta", "b"),
            make_chunk("gamma", "c"),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    )
    results = store.query([0.9, 0.1, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "a"


def test_top_k_limits():
    store = make_store()
    store.add(
        [make_chunk(f"chunk {i}", f"id{i}") for i in range(5)],
        [[float(i), 0.0, 0.0] for i in range(5)],
    )
    results = store.query([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2


def test_metadata_roundtrip():
    store = make_store()
    store.add(
        [make_chunk("text", "id1", page=7, source="ml.pdf")],
        [[1.0, 0.0, 0.0]],
    )
    results = store.query([1.0, 0.0, 0.0], top_k=1)
    assert results[0].chunk.chunk_id == "id1"
    assert results[0].chunk.text == "text"
    assert results[0].chunk.page_number == 7
    assert results[0].chunk.source_file == "ml.pdf"


def test_score_range():
    store = make_store()
    store.add([make_chunk("a", "a")], [[1.0, 0.0, 0.0]])
    results = store.query([1.0, 0.0, 0.0], top_k=1)
    assert 0.0 <= results[0].score <= 1.0
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_empty_store_returns_empty():
    store = make_store()
    assert store.query([1.0, 0.0, 0.0]) == []


def test_add_duplicate_ids_raises():
    store = make_store()
    chunk = make_chunk("text", "dup_id")
    store.add([chunk], [[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="already in collection"):
        store.add([chunk], [[1.0, 0.0, 0.0]])


def test_add_duplicate_ids_within_batch_raises():
    store = make_store()
    chunk = make_chunk("text", "dup_id")
    with pytest.raises(ValueError, match="within the batch"):
        store.add([chunk, chunk], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_add_count_mismatch_raises():
    store = make_store()
    with pytest.raises(ValueError, match="length mismatch"):
        store.add([make_chunk("a", "a")], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
