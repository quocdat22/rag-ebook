"""Unit tests for src.retrieval.retriever — fake embedder + store, no real services."""

from src.chunking.splitter import Chunk
from src.retrieval.retriever import retrieve
from src.vectorstore.chroma_store import RetrievedChunk


class FakeEmbedder:
    """Records calls; returns a fixed vector for every input."""

    def __init__(self, vector: list[float]):
        self.vector = vector
        self.calls: list[object] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self.vector] * len(texts)

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.vector


class FakeStore:
    """Records calls; returns a fixed result list (respecting top_k)."""

    def __init__(self, results: list[RetrievedChunk]):
        self.results = results
        self.calls: list[tuple[list[float], int]] = []

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        self.calls.append((query_embedding, top_k))
        return self.results[:top_k]

    def add(self, *args, **kwargs) -> None:
        pass

    def delete_by_source(self, source_file: str) -> int:
        return 0


def make_chunk(chunk_id: str = "c", score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            text=f"text of {chunk_id}",
            page_number=1,
            source_file="book.pdf",
        ),
        score=score,
    )


def test_calls_embedder_then_store():
    embedder = FakeEmbedder([0.1, 0.2])
    store = FakeStore([make_chunk()])
    retrieve("the question", embedder, store, top_k=5)
    # embed_query called first with the raw question (instruction prefix handled inside).
    assert embedder.calls == ["the question"]
    # store.query receives the query embedding produced above and top_k=5.
    assert store.calls[0][0] == [0.1, 0.2]
    assert store.calls[0][1] == 5


def test_question_embedded_exactly_once():
    embedder = FakeEmbedder([0.1])
    store = FakeStore([make_chunk(), make_chunk("c2", 0.8)])
    retrieve("question?", embedder, store, top_k=2)
    assert embedder.calls.count("question?") == 1


def test_returns_expected_count():
    embedder = FakeEmbedder([0.1])
    store = FakeStore([make_chunk(f"c{i}", score) for i, score in enumerate([0.9, 0.8, 0.7, 0.6])])
    results = retrieve("q", embedder, store, top_k=3)
    assert len(results) == 3  # min(top_k, store size)


def test_top_k_passed_through():
    embedder = FakeEmbedder([0.1])
    store = FakeStore([make_chunk(f"c{i}") for i in range(5)])
    retrieve("q", embedder, store, top_k=3)
    assert store.calls[0][1] == 3


def test_min_score_filters():
    embedder = FakeEmbedder([0.1])
    store = FakeStore([make_chunk(f"c{score}", score) for score in [0.9, 0.2, 0.1]])
    results = retrieve("q", embedder, store, top_k=5, min_score=0.5)
    assert len(results) == 1
    assert results[0].score == 0.9


def test_min_score_zero_keeps_all():
    embedder = FakeEmbedder([0.1])
    store = FakeStore([make_chunk(f"c{score}", score) for score in [0.3, 0.2, 0.1]])
    results = retrieve("q", embedder, store, top_k=5, min_score=0.0)
    assert len(results) == 3


def test_empty_store_ok():
    embedder = FakeEmbedder([0.1])
    store = FakeStore([])
    assert retrieve("q", embedder, store) == []


def test_sort_by_score():
    embedder = FakeEmbedder([0.1])
    # Store returns results in arbitrary order.
    store = FakeStore([make_chunk("low", 0.3), make_chunk("high", 0.9), make_chunk("mid", 0.6)])
    results = retrieve("q", embedder, store, top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].chunk.chunk_id == "high"
