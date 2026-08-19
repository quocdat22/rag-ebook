"""Unit tests for src.chunking.splitter."""

from src.chunking.splitter import (
    estimate_tokens,
    pack_segments,
    split_documents,
    split_into_segments,
)
from src.ingestion.pdf_loader import Document

PARA = (
    "Embeddings map objects into a vector space where similar items are close "
    "together, which enables fast similarity search over large collections of "
    "technical documentation and source code."
)

CODE_BLOCK = "```python\n" + "\n".join(f"x{i} = {i} * 2" for i in range(12)) + "\n```"


def make_doc(text: str, page: int = 1, source: str = "book.pdf") -> Document:
    return Document(text=text, page_number=page, source_file=source)


def test_chunks_within_size():
    text = "\n\n".join(PARA for _ in range(25))
    chunks = split_documents([make_doc(text)], chunk_size=700, overlap=100)
    assert chunks
    assert all(estimate_tokens(c.text) <= 700 for c in chunks)


def test_code_block_not_split():
    text = f"Introduction paragraph.\n\n{CODE_BLOCK}\n\nClosing paragraph."
    chunks = split_documents([make_doc(text)], chunk_size=700, overlap=100)
    assert any("```" in c.text for c in chunks)
    for c in chunks:
        if "```" in c.text:
            assert c.text.count("```") % 2 == 0


def test_overlap():
    chunk_size, overlap = 200, 50
    segments = [f"p{i} " + "word " * 30 for i in range(30)]
    packed = pack_segments(segments, chunk_size=chunk_size, overlap=overlap)
    assert len(packed) >= 2
    head, tail = packed[0], packed[1]
    shared = 0
    while shared < len(head) and shared < len(tail) and head[-1 - shared] == tail[shared]:
        shared += 1
    assert shared >= 1
    carried_tokens = sum(estimate_tokens(s) for s in tail[:shared])
    assert carried_tokens >= overlap // 2


def test_metadata_preserved():
    docs = [
        make_doc("\n\n".join(PARA for _ in range(12)), page=1, source="a.pdf"),
        make_doc("\n\n".join(PARA for _ in range(12)), page=2, source="a.pdf"),
    ]
    chunks = split_documents(docs)
    assert all(c.source_file == "a.pdf" for c in chunks)
    assert {c.page_number for c in chunks} == {1, 2}


def test_chunk_ids_unique():
    text = "\n\n".join(PARA for _ in range(25))
    chunks = split_documents([make_doc(text)])
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_input():
    assert split_documents([], chunk_size=700, overlap=100) == []


def test_single_long_segment_split_at_newline():
    code = "```python\n" + "\n".join(f"line_{i} = 'x' * 10" for i in range(200)) + "\n```"
    chunks = split_documents([make_doc(code)], chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    joined = "\n".join(c.text for c in chunks)
    assert all(f"line_{i}" in joined for i in range(200))


def test_split_into_segments_keeps_code_whole():
    text = f"Intro.\n\n{CODE_BLOCK}\n\nOutro."
    segments = split_into_segments(text)
    assert any(seg.startswith("```") and seg.endswith("```") for seg in segments)
