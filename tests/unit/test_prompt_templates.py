"""Unit tests for src.generation.prompt_templates."""

from src.chunking.splitter import Chunk
from src.generation.prompt_templates import (
    SYSTEM_PROMPT,
    build_user_prompt,
    extract_cited_indices,
)
from src.vectorstore.chroma_store import RetrievedChunk


def make_chunk(text: str, page: int = 1, source: str = "sample_tech_ebook.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"{source}::p{page}::0",
            text=text,
            page_number=page,
            source_file=source,
        ),
        score=0.9,
    )


def test_user_prompt_contains_context_and_question():
    chunks = [
        make_chunk("alpha context", page=2),
        make_chunk("beta context", page=3),
    ]
    prompt = build_user_prompt(chunks, "What is a vector database?")
    assert "What is a vector database?" in prompt
    assert "alpha context" in prompt
    assert "beta context" in prompt
    assert "(sample_tech_ebook.pdf, page 2)" in prompt
    assert "(sample_tech_ebook.pdf, page 3)" in prompt
    # Question comes after the context, and the final line is the citation instruction.
    assert prompt.index("Question:") > prompt.index("beta context")
    assert prompt.rstrip().endswith("Answer with inline citations like [1].")


def test_numbering_starts_at_1():
    chunks = [make_chunk("a", 1), make_chunk("b", 2), make_chunk("c", 3)]
    prompt = build_user_prompt(chunks, "q")
    assert "[1] (sample_tech_ebook.pdf, page 1)" in prompt
    assert "[2] (sample_tech_ebook.pdf, page 2)" in prompt
    assert "[3] (sample_tech_ebook.pdf, page 3)" in prompt
    assert prompt.index("[1]") < prompt.index("[2]") < prompt.index("[3]")


def test_system_prompt_is_constant():
    assert isinstance(SYSTEM_PROMPT, str)
    assert "cite sources inline using [n]" in SYSTEM_PROMPT


def test_extract_citations():
    assert extract_cited_indices("See [1] and [2][3].") == [1, 2, 3]
    assert extract_cited_indices("No citations here") == []
    # Out-of-range indices are kept; range validation happens in Phase 5.
    assert extract_cited_indices("See [99]") == [99]
    assert extract_cited_indices("See [1] and [1] again") == [1]  # deduplicated
