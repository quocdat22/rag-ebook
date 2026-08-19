"""Index pipeline: PDF file -> chunks -> embeddings -> vector store (SPEC 5.7).

Runs synchronously (ingesting a large PDF makes the request slow — accepted
for the MVP; a background task + status polling is a v2 idea, SPEC mục 10).
"""

import logging
import time
from pathlib import Path

from src.chunking.splitter import split_documents
from src.embedding.ollama_client import EmbeddingClient
from src.ingestion.pdf_loader import load_pdf
from src.vectorstore.chroma_store import VectorStore

logger = logging.getLogger(__name__)


class IndexPipeline:
    def __init__(
        self,
        embedder: EmbeddingClient,
        store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def run(self, pdf_path: str) -> int:
        """Index one PDF into the store; returns the number of chunks indexed.

        Idempotent: chunks of the same ``source_file`` are replaced, not
        duplicated. Old chunks are deleted only *after* the new embeddings
        succeeded, so a failure mid-embedding leaves the previous index intact.
        A PDF with no extractable content raises and writes nothing.
        """
        t0 = time.perf_counter()
        docs = load_pdf(pdf_path)
        chunks = split_documents(docs, chunk_size=self._chunk_size, overlap=self._chunk_overlap)
        if not chunks:
            raise ValueError(f"No chunks produced from PDF: {pdf_path}")
        logger.info(
            "Parsed %d pages -> %d chunks from %s",
            len(docs),
            len(chunks),
            Path(pdf_path).name,
        )

        embeddings = self._embedder.embed([c.text for c in chunks])
        source_file = chunks[0].source_file
        removed = self._store.delete_by_source(source_file)
        self._store.add(chunks, embeddings)
        logger.info(
            "Indexed %d chunks (removed %d stale) in %.1fs",
            len(chunks),
            removed,
            time.perf_counter() - t0,
        )
        return len(chunks)
