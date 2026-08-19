"""Unit tests for the two pipelines — fakes only, no real services."""

import pymupdf
import pytest

from src.chunking.splitter import Chunk
from src.ingestion.pdf_loader import EmptyDocumentError
from src.pipeline.index_pipeline import IndexPipeline
from src.pipeline.query_pipeline import QueryPipeline
from src.vectorstore.chroma_store import RetrievedChunk


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 4] * len(texts)

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 4


class FakeStore:
    def __init__(self, results: list[RetrievedChunk] | None = None):
        self.results = results or []
        self.add_calls = 0
        self.deleted_sources: list[str] = []

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        return self.results[:top_k]

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.add_calls += 1

    def delete_by_source(self, source_file: str) -> int:
        self.deleted_sources.append(source_file)
        return 0


class FakeLLM:
    def __init__(self, answer: str):
        self.answer = answer

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.answer


def make_retrieved(chunk_id: str, text: str, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, text=text, page_number=page, source_file="b.pdf"),
        score=0.9,
    )


def test_query_pipeline_filters_out_of_range_citations():
    chunks = [make_retrieved("a::p1::0", "alpha"), make_retrieved("b::p2::1", "beta")]
    store = FakeStore(chunks)
    llm = FakeLLM("Answer uses [1] and [99] and [2].")
    pipeline = QueryPipeline(FakeEmbedder(), store, llm, top_k=5)
    result = pipeline.run("question")

    assert result.answer == "Answer uses [1] and [99] and [2]."
    # [99] is out of range (only 2 chunks) -> dropped, not a crash.
    assert [c.chunk_id for c in result.citations] == ["a::p1::0", "b::p2::1"]
    assert len(result.used_chunks) == 2
    assert result.citations[0].page_number == 1
    assert result.citations[0].source_file == "b.pdf"


def test_query_pipeline_empty_context():
    store = FakeStore([])  # nothing retrieved
    llm = FakeLLM("I don't know [1].")
    pipeline = QueryPipeline(FakeEmbedder(), store, llm, top_k=5)
    result = pipeline.run("question")
    assert result.citations == []
    assert result.used_chunks == []


def test_index_pipeline_empty_pdf_raises_without_adding(tmp_path):
    blank = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()  # a page with no text at all
    doc.save(str(blank))
    doc.close()

    store = FakeStore()
    pipeline = IndexPipeline(FakeEmbedder(), store, chunk_size=700, chunk_overlap=100)
    with pytest.raises(EmptyDocumentError):
        pipeline.run(str(blank))
    assert store.add_calls == 0
    assert store.deleted_sources == []
