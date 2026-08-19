"""Retrieval: combine embedding + vector store into one `question -> top-k` step.

Pure function with dependencies passed as parameters (SPEC 5.5, F4) — the
easiest shape to test with fakes. If caching/re-ranking is added later, wrap it
in a `Retriever` class with the same dependency-injection style.
"""

from src.embedding.ollama_client import EmbeddingClient
from src.vectorstore.chroma_store import RetrievedChunk, VectorStore


def retrieve(
    question: str,
    embedder: EmbeddingClient,
    store: VectorStore,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    """Retrieve the top-k chunks most similar to `question`.

    Flow (SPEC 4.2):
    1. ``embedder.embed_query(question)`` — the instruction prefix matters for
       the instruction-aware ``qwen3-embedding:0.6b`` model (SPEC 5.3); never
       use plain ``embed()`` for queries.
    2. ``store.query(embedding, top_k=top_k)``.
    3. If ``min_score > 0``, keep only results with ``score >= min_score`` — a
       guard against feeding irrelevant context into the prompt.
    4. Return sorted by score descending (we sort ourselves rather than
       trusting the store's ordering).

    Returns an empty list for an empty store — never raises.
    """
    query_embedding = embedder.embed_query(question)
    results = store.query(query_embedding, top_k=top_k)
    if min_score > 0:
        results = [r for r in results if r.score >= min_score]
    return sorted(results, key=lambda r: r.score, reverse=True)
