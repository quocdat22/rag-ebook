"""Smoke test for Phase 3: retrieve top-k chunks for a real question (Ollama + Chroma).

Run:
    uv run python scripts/smoke_retrieve.py "your question" [--collection NAME] [--top-k 5] [--min-score 0.0]

Expects: prints the top-k chunks (page, source, score, first ~100 chars) from the
collection indexed in Phase 2 (`scripts/smoke_index.py`).
"""

import argparse
import sys
from pathlib import Path

# Make `src.*` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb

from src.config import Settings
from src.embedding.ollama_client import OllamaEmbeddingClient, OllamaUnavailableError
from src.retrieval.retriever import retrieve
from src.vectorstore.chroma_store import ChromaVectorStore


def resolve_collection(persist_dir: str, name: str | None) -> str:
    """Pick the requested collection, or the only one available in the store."""
    client = chromadb.PersistentClient(path=persist_dir)
    names = [collection.name for collection in client.list_collections()]
    if name:
        if name not in names:
            print(
                f"ERROR: collection '{name}' not found in {persist_dir}.\nAvailable: {names}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return name
    if len(names) == 1:
        return names[0]
    if not names:
        print(
            "ERROR: no collections found — run scripts/smoke_index.py first.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(
        f"ERROR: multiple collections {names} — pass --collection NAME.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    settings = Settings()
    collection_name = resolve_collection(settings.chroma_persist_dir, args.collection)
    store = ChromaVectorStore(
        collection_name=collection_name, persist_dir=settings.chroma_persist_dir
    )

    print(
        f"Question: {args.question!r}  (collection: {collection_name}, "
        f"top_k={args.top_k}, min_score={args.min_score})"
    )
    try:
        with OllamaEmbeddingClient(
            model=settings.ollama_embed_model, host=settings.ollama_host
        ) as embedder:
            results = retrieve(
                args.question, embedder, store, top_k=args.top_k, min_score=args.min_score
            )
    except OllamaUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No results.")
        return 0
    for index, item in enumerate(results, 1):
        chunk = item.chunk
        print(
            f"{index}. score={item.score:.4f} page={chunk.page_number} source={chunk.source_file}"
        )
        print(f"   {chunk.text[:100]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
