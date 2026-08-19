"""Prompt templates for RAG generation (SPEC 5.6).

The system prompt is a module-level constant and MUST NOT change between
requests — DeepSeek's prefix-cache pricing makes the repeated system prompt
much cheaper. Anything that varies per question lives in the user prompt.
"""

import re

from src.vectorstore.chroma_store import RetrievedChunk

SYSTEM_PROMPT: str = (
    "You are a technical assistant that answers questions strictly based on the "
    "provided context. If the context does not contain the answer, say so explicitly. "
    "Always cite sources inline using [n] where n is the context number, e.g. [1] or [2][3]. "
    "Never invent sources."
)

CITATION_RE = re.compile(r"\[(\d+)\]")


def build_user_prompt(context_chunks: list[RetrievedChunk], question: str) -> str:
    """Build the user prompt: numbered context (starting at [1]) + question at the end.

    Format::

        Context:
        [1] (sample_tech_ebook.pdf, page 2)
        <chunk text>

        [2] (sample_tech_ebook.pdf, page 3)
        <chunk text>

        Question: What is a vector database?

        Answer with inline citations like [1].
    """
    lines = ["Context:"]
    for index, item in enumerate(context_chunks, start=1):
        chunk = item.chunk
        lines.append(f"[{index}] ({chunk.source_file}, page {chunk.page_number})")
        lines.append(chunk.text)
        lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Answer with inline citations like [1].")
    return "\n".join(lines)


def extract_cited_indices(answer: str) -> list[int]:
    """Context indices cited in `answer`, in order of appearance, deduplicated.

    e.g. ``"See [1] and [2][3]."`` -> ``[1, 2, 3]``. Out-of-range indices are
    kept as-is (validated against the context length in Phase 5).
    """
    indices: list[int] = []
    for match in CITATION_RE.finditer(answer):
        index = int(match.group(1))
        if index not in indices:
            indices.append(index)
    return indices
