"""Demo script for the report: a few sample questions with answers + citations.

Runs against the real `rag_ebook` collection (Ollama + DeepSeek). If the
collection is empty (fresh clone), it first indexes the committed fixture PDF
so the demo works out of the box. Each run costs a little DeepSeek money.

Run: uv run python scripts/demo.py
"""

import sys
from pathlib import Path

# Make `src.*` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.embedding.ollama_client import OllamaEmbeddingClient
from src.errors import ConfigurationError, RagEbookError
from src.generation.deepseek_client import DeepSeekClient
from src.logging_config import setup_logging
from src.pipeline.index_pipeline import IndexPipeline
from src.pipeline.query_pipeline import QueryPipeline
from src.vectorstore.chroma_store import ChromaVectorStore

FIXTURE_PDF = Path("tests/fixtures/sample_tech_ebook.pdf")
DEFAULT_COLLECTION = "rag_ebook"

DEMO_QUESTIONS = [
    "What is a vector database?",
    "How does a RAG pipeline work?",
    "Why is cosine similarity popular for text embeddings?",
    "What are embeddings and what are they used for?",
]


def main() -> int:
    setup_logging()
    settings = Settings()
    store = ChromaVectorStore(
        collection_name=DEFAULT_COLLECTION, persist_dir=settings.chroma_persist_dir
    )
    embedder = OllamaEmbeddingClient(model=settings.ollama_embed_model, host=settings.ollama_host)
    try:
        llm = DeepSeekClient(
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            base_url=settings.deepseek_base_url,
        )
    except ConfigurationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if store.count() == 0:
        print("Collection is empty — indexing the fixture PDF first...")
        index_pipeline = IndexPipeline(embedder, store, settings.chunk_size, settings.chunk_overlap)
        index_pipeline.run(str(FIXTURE_PDF))
    print(f"Collection '{DEFAULT_COLLECTION}': {store.count()} chunks\n")

    query_pipeline = QueryPipeline(embedder, store, llm, settings.top_k)
    try:
        for question in DEMO_QUESTIONS:
            print("=" * 70)
            print(f"Q: {question}")
            result = query_pipeline.run(question)
            print(f"A: {result.answer}\n")
            for citation in result.citations:
                print(
                    f"  📄 {citation.source_file} — p.{citation.page_number} ({citation.chunk_id})"
                )
            print()
    except RagEbookError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
