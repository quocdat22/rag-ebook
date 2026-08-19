"""ChromaDB vector store: persist chunk embeddings locally, query by cosine.

The collection is pinned to cosine space (``hnsw:space: cosine``) — the
standard for text RAG. Retrieved scores are cosine similarities computed as
``score = 1 - distance``; this only holds for cosine, so do not switch the
space to L2/IP without updating the score conversion here.

``persist_dir=None`` uses an in-memory ``EphemeralClient`` (tests); a real path
persists to disk. Changing the embedding model (different dimension) requires
a full re-index — the store does not silently mix dimensions.
"""

from typing import Any, Protocol, cast

import chromadb
from pydantic import BaseModel

from src.chunking.splitter import Chunk


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float  # cosine similarity: higher = more similar


def _duplicates(ids: list[str]) -> list[str]:
    """Ids that appear more than once, in first-seen order."""
    seen: set[str] = set()
    dupes: list[str] = []
    for chunk_id in ids:
        if chunk_id in seen and chunk_id not in dupes:
            dupes.append(chunk_id)
        seen.add(chunk_id)
    return dupes


class VectorStore(Protocol):
    """Vector store contract (SPEC 5.4) — swap-in-able / mockable."""

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]: ...

    def delete_by_source(self, source_file: str) -> int: ...


class ChromaVectorStore:
    """Chroma-backed vector store with cosine similarity."""

    def __init__(self, collection_name: str, persist_dir: str | None = None) -> None:
        if persist_dir is None:
            self._client: chromadb.ClientAPI = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    @property
    def collection_name(self) -> str:
        return self._collection.name

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks with their precomputed embeddings (lengths must match).

        Duplicate chunk_ids (within the batch or already in the collection) are
        rejected loudly — Chroma would otherwise silently skip them (first-write
        wins), which is surprising. Re-index from a fresh collection instead.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate chunk_id(s) within the batch: {_duplicates(ids)}")
        existing = self._collection.get(ids=ids, include=[])["ids"]
        if existing:
            raise ValueError(
                f"chunk_id(s) already in collection '{self._collection.name}': {existing[:5]}"
            )
        self._collection.add(
            ids=ids,
            documents=[c.text for c in chunks],
            embeddings=cast(Any, embeddings),  # chromadb stubs reject list[list[float]]
            metadatas=[
                {"page_number": c.page_number, "source_file": c.source_file} for c in chunks
            ],
        )

    def delete_by_source(self, source_file: str) -> int:
        """Delete every chunk of one source file; returns the number removed."""
        ids = self._collection.get(where={"source_file": source_file}, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        """Total number of stored chunks."""
        return self._collection.count()

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top-k most similar chunks with cosine similarity scores."""
        if self._collection.count() == 0:
            return []
        result: dict[str, Any] = cast(
            Any,
            self._collection.query(
                query_embeddings=cast(Any, [query_embedding]),
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            ),
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            retrieved.append(
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id=chunk_id,
                        text=text,
                        page_number=int(meta["page_number"]),
                        source_file=meta["source_file"],
                    ),
                    score=1.0 - float(distance),
                )
            )
        return retrieved
