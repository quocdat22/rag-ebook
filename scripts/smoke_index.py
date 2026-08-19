"""Smoke test for Phase 2: index one real PDF end-to-end with real Ollama.

Pipeline: load_pdf -> split_documents -> OllamaEmbeddingClient.embed -> ChromaVectorStore.add.
Expects: prints per-batch progress and the final chunk count; `data/chroma/` appears.

Run:
    uv run python scripts/smoke_index.py [path-to.pdf] [--collection NAME]

Re-running is idempotent: the target collection is dropped and re-created, so
the store's duplicate-id guard never trips.
"""

import argparse
import re
import sys
import time
from pathlib import Path

# Make `src.*` importable when run as a plain script (`uv run python scripts/smoke_index.py`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.splitter import split_documents
from src.config import Settings
from src.embedding.ollama_client import OllamaEmbeddingClient, OllamaUnavailableError
from src.ingestion.pdf_loader import load_pdf
from src.vectorstore.chroma_store import ChromaVectorStore

DEFAULT_PDF = Path(
    "data/ebooks/"
    "Designing Machine Learning Systems An Iterative Process for "
    "Production-Ready (Chip Huyen)[21-43].pdf"
)
DEFAULT_BATCH_SIZE = 8


def slugify(name: str) -> str:
    """Turn a file stem into a valid Chroma collection name (3-512 [a-zA-Z0-9._-])."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return slug if len(slug) >= 3 else f"book-{slug}"


def reset_collection(collection_name: str, persist_dir: str) -> None:
    """Drop the target collection so re-runs start from a clean slate."""
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(collection_name)
        print(f"  reset existing collection '{collection_name}'")
    except chromadb.errors.NotFoundError:
        pass  # fresh collection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", nargs="?", default=str(DEFAULT_PDF))
    parser.add_argument(
        "--collection",
        default=None,
        help="Chroma collection name (default: derived from the PDF filename)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    settings = Settings()
    collection_name = args.collection or slugify(Path(args.pdf_path).stem)

    try:
        t0 = time.perf_counter()
        docs = load_pdf(args.pdf_path)
        chunks = split_documents(
            docs, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
        )
        print(f"Parsed {len(docs)} pages -> {len(chunks)} chunks ({time.perf_counter() - t0:.1f}s)")
        if not chunks:
            print("ERROR: no chunks produced; nothing to index.", file=sys.stderr)
            return 1

        reset_collection(collection_name, settings.chroma_persist_dir)
        store = ChromaVectorStore(
            collection_name=collection_name, persist_dir=settings.chroma_persist_dir
        )

        embedder = OllamaEmbeddingClient(
            model=settings.ollama_embed_model, host=settings.ollama_host
        )
        try:
            for start in range(0, len(chunks), args.batch_size):
                batch = chunks[start : start + args.batch_size]
                t1 = time.perf_counter()
                embeddings = embedder.embed([c.text for c in batch])
                store.add(batch, embeddings)
                print(
                    f"  indexed {start + len(batch)}/{len(chunks)} chunks "
                    f"(+{time.perf_counter() - t1:.1f}s)",
                    flush=True,
                )
        finally:
            embedder.close()

        print(
            f"Done: {len(chunks)} chunks indexed into collection "
            f"'{collection_name}' at {settings.chroma_persist_dir}"
        )
        return 0
    except OllamaUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
