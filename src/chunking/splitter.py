"""Code-block-aware text chunking (custom splitter, no LangChain).

Strategy: split each page into *atomic* segments (fenced code blocks, heading
lines, paragraphs), then greedily pack segments into chunks of at most
`chunk_size` estimated tokens. Segments are never cut in half, so code blocks
and headings stay intact — the only exception is a single segment larger than
`chunk_size` (e.g. a very long code block), which is split at newlines so no
content is lost.
"""

import re

from pydantic import BaseModel

from src.ingestion.pdf_loader import Document


class Chunk(BaseModel):
    chunk_id: str
    text: str
    page_number: int
    source_file: str


# Atomic-segment patterns
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
HEADING_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
BLANK_LINE_RE = re.compile(r"\n\s*\n")

# A plain paragraph longer than this (in estimated tokens) is split at newlines.
MAX_PLAIN_TOKENS = 1200
# Token estimate multiplier for technical text (many symbols/long identifiers).
TOKENS_PER_WORD = 1.3


def estimate_tokens(text: str) -> int:
    """Rough token estimate: word count * 1.3.

    Deliberately crude for the MVP (no tokenizer dependency); the multiplier
    over-weights technical text with many symbols. Isolated here so a real
    tokenizer can be swapped in later.
    """
    return int(len(text.split()) * TOKENS_PER_WORD)


def split_into_segments(text: str) -> list[str]:
    """Split page text into atomic segments.

    Fenced code blocks (````` ``` ... ``` `````) become single unbreakable
    segments; the remaining text is split into heading lines and paragraphs.
    """
    segments: list[str] = []
    pos = 0
    for match in FENCE_RE.finditer(text):
        if match.start() > pos:
            segments.extend(_split_plain(text[pos : match.start()]))
        segments.append(match.group(0).strip())
        pos = match.end()
    if pos < len(text):
        segments.extend(_split_plain(text[pos:]))
    return [seg for seg in segments if seg.strip()]


def _split_plain(text: str) -> list[str]:
    """Split non-fenced text into heading lines and paragraphs."""
    segments: list[str] = []
    for block in BLANK_LINE_RE.split(text):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        if HEADING_RE.match(lines[0]):
            segments.append(lines[0])
            rest = "\n".join(lines[1:]).strip()
            if rest:
                segments.extend(_split_long(rest))
        else:
            segments.extend(_split_long(block))
    return segments


def _split_long(text: str) -> list[str]:
    """Split a paragraph at newlines if it exceeds `MAX_PLAIN_TOKENS`."""
    if estimate_tokens(text) <= MAX_PLAIN_TOKENS:
        return [text]
    return [line for line in text.split("\n") if line.strip()]


def carry_over_tail(segments: list[str], overlap: int) -> list[str]:
    """Tail of `segments` totaling roughly `overlap` tokens, to seed the next chunk."""
    tail: list[str] = []
    tokens = 0
    for seg in reversed(segments):
        seg_tokens = estimate_tokens(seg)
        if tokens and tokens + seg_tokens > overlap:
            break
        tail.append(seg)
        tokens += seg_tokens
    tail.reverse()
    return tail


def split_oversized(seg: str, chunk_size: int) -> list[str]:
    """Split one oversized segment (e.g. a very long code block) at newlines."""
    pieces: list[str] = []
    current = ""
    for line in seg.split("\n"):
        candidate = line if not current else f"{current}\n{line}"
        if current and estimate_tokens(candidate) > chunk_size:
            pieces.append(current)
            current = line
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def pack_segments(
    segments: list[str], chunk_size: int = 700, overlap: int = 100
) -> list[list[str]]:
    """Greedily pack atomic segments into chunks of at most `chunk_size` tokens.

    Segments are never split in half, except a single segment larger than
    `chunk_size` (very long code block), which is split at newlines. When a
    chunk closes, its tail (~`overlap` tokens) is carried into the next chunk.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        chunks.append(current)
        current = carry_over_tail(current, overlap)
        current_tokens = estimate_tokens("\n\n".join(current))

    def add(seg: str) -> None:
        nonlocal current, current_tokens
        seg_tokens = estimate_tokens(seg)
        if seg_tokens > chunk_size:
            flush()
            pieces = split_oversized(seg, chunk_size)
            if not pieces:
                return
            if len(pieces) == 1:
                # Single un-splittable line: keep as its own (oversized) chunk.
                current, current_tokens = [pieces[0]], estimate_tokens(pieces[0])
            else:
                for piece in pieces:
                    add(piece)
            return
        if current and estimate_tokens("\n\n".join(current + [seg])) > chunk_size:
            flush()
        if current and estimate_tokens("\n\n".join(current + [seg])) > chunk_size:
            # Carried tail too large to share with this segment — drop the carry.
            current, current_tokens = [], 0
        current.append(seg)
        current_tokens = estimate_tokens("\n\n".join(current))

    for seg in segments:
        add(seg)
    flush()
    return chunks


def split_documents(docs: list[Document], chunk_size: int = 700, overlap: int = 100) -> list[Chunk]:
    """Split `Document`s into code-block-aware `Chunk`s, preserving metadata."""
    chunks: list[Chunk] = []
    index = 0
    for doc in docs:
        segments = split_into_segments(doc.text)
        for packed in pack_segments(segments, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source_file}::p{doc.page_number}::{index}",
                    text="\n\n".join(packed),
                    page_number=doc.page_number,
                    source_file=doc.source_file,
                )
            )
            index += 1
    return chunks
