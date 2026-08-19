"""End-to-end pipeline test: real fixture PDF + real Chroma (temp dir), fake embedder + fake LLM.

Runs anywhere (even CI) without Ollama/DeepSeek — the embedding and LLM
dependencies are deterministic fakes (SPEC mục 6). Marked ``integration`` so it
can be excluded from a fast unit-only run.
"""

import hashlib

import pytest

from src.pipeline.index_pipeline import IndexPipeline
from src.pipeline.query_pipeline import QueryPipeline
from src.vectorstore.chroma_store import ChromaVectorStore

FIXTURE_PDF = "tests/fixtures/sample_tech_ebook.pdf"


class FakeEmbedder:
    """Deterministic pseudo-embeddings: sha256 of the text -> 8-dim vector."""

    DIM = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        return [byte / 255.0 for byte in hashlib.sha256(text.encode()).digest()[: self.DIM]]


class FakeLLM:
    def __init__(self, answer: str = "The answer is in the context [1]."):
        self.answer = answer
        self.last_user_prompt: str | None = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.last_user_prompt = user_prompt
        return self.answer


def make_store(tmp_path) -> ChromaVectorStore:
    return ChromaVectorStore(collection_name="e2e_test", persist_dir=str(tmp_path))


@pytest.mark.integration
def test_index_then_query_full_flow(tmp_path):
    store = make_store(tmp_path)
    embedder = FakeEmbedder()
    llm = FakeLLM()

    index_pipeline = IndexPipeline(embedder, store, chunk_size=700, chunk_overlap=100)
    chunks_indexed = index_pipeline.run(FIXTURE_PDF)
    assert chunks_indexed > 0
    assert store.count() == chunks_indexed

    query_pipeline = QueryPipeline(embedder, store, llm, top_k=3)
    result = query_pipeline.run("What is a vector database?")
    assert result.answer
    assert result.used_chunks
    assert llm.last_user_prompt  # prompt was built from the retrieved context

    used_ids = {r.chunk.chunk_id for r in result.used_chunks}
    assert all(c.chunk_id in used_ids for c in result.citations)
    for citation in result.citations:
        assert citation.source_file == "sample_tech_ebook.pdf"
        assert isinstance(citation.page_number, int) and citation.page_number >= 1
        assert citation.text


@pytest.mark.integration
def test_reindex_same_file_no_duplicates(tmp_path):
    store = make_store(tmp_path)
    pipeline = IndexPipeline(FakeEmbedder(), store, chunk_size=700, chunk_overlap=100)
    first = pipeline.run(FIXTURE_PDF)
    second = pipeline.run(FIXTURE_PDF)
    assert second == first
    # Re-indexing replaces, never duplicates.
    assert store.count() == first
