"""Smoke test for Phase 4: one real DeepSeek API call (costs a little money).

Builds a small hand-written context about the fixture PDF plus one question and
prints the generated answer. Expects an English answer with inline citations
like [n] and a non-empty list of cited indices.

Run: uv run python scripts/smoke_generate.py
"""

import sys
from pathlib import Path

# Make `src.*` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.splitter import Chunk
from src.config import Settings
from src.generation.deepseek_client import DeepSeekClient, GenerationError
from src.generation.prompt_templates import SYSTEM_PROMPT, build_user_prompt, extract_cited_indices
from src.vectorstore.chroma_store import RetrievedChunk


def fake_context() -> list[RetrievedChunk]:
    """A couple of hand-written lines standing in for retrieved chunks."""
    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id="fixture::p1::0",
                text=(
                    "A vector database stores high-dimensional vectors and supports fast "
                    "similarity search. Instead of exact keyword matching, retrieval is "
                    "based on distance between embeddings, so systems find semantically "
                    "related items even when they share no common terms."
                ),
                page_number=1,
                source_file="sample_tech_ebook.pdf",
            ),
            score=0.9,
        ),
        RetrievedChunk(
            chunk=Chunk(
                chunk_id="fixture::p2::1",
                text=(
                    "A typical RAG pipeline first indexes a corpus by embedding every chunk "
                    "of text and storing the vectors, then answers a question by embedding "
                    "the query and searching for the most similar stored vectors. The "
                    "retrieved passages are handed to a language model as context for "
                    "generating the answer."
                ),
                page_number=2,
                source_file="sample_tech_ebook.pdf",
            ),
            score=0.85,
        ),
    ]


def main() -> int:
    settings = Settings()
    if not settings.deepseek_api_key:
        print("ERROR: DEEPSEEK_API_KEY is not set in .env", file=sys.stderr)
        return 1

    question = "What is a vector database and how does a RAG pipeline use it?"
    user_prompt = build_user_prompt(fake_context(), question)
    client = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        base_url=settings.deepseek_base_url,
    )
    try:
        answer = client.generate(SYSTEM_PROMPT, user_prompt)
    except GenerationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Question: {question}")
    print(f"Cited indices: {extract_cited_indices(answer)}")
    print("--- answer ---")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
