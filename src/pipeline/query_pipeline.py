"""Query pipeline: question -> retrieved context -> LLM answer + citations (SPEC 5.7).

Follows SPEC 4.2: retrieve top-k chunks (with an optional ``min_score``
similarity threshold against irrelevant context), build the user prompt, call
the LLM, then map the ``[n]`` citations in the answer back to the actual
sources. Retrieval params come from ``Settings`` and can be overridden per call.
"""

import logging
import time

from pydantic import BaseModel

from src.embedding.ollama_client import EmbeddingClient
from src.generation.deepseek_client import LLMClient
from src.generation.prompt_templates import SYSTEM_PROMPT, build_user_prompt, extract_cited_indices
from src.retrieval.retriever import retrieve
from src.vectorstore.chroma_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int
    text: str


class AnswerResult(BaseModel):
    answer: str
    citations: list[Citation]  # sources actually cited as [n] in the answer
    used_chunks: list[RetrievedChunk]  # the full context that went into the prompt


class QueryPipeline:
    def __init__(
        self,
        embedder: EmbeddingClient,
        store: VectorStore,
        llm: LLMClient,
        top_k: int,
        min_score: float,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._top_k = top_k
        self._min_score = min_score

    def run(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> AnswerResult:
        """Answer `question`; optional per-call overrides for retrieval params.

        ``top_k`` / ``min_score`` default to the values given at construction
        (which come from ``Settings``); pass ``None`` to keep those defaults.
        """
        t0 = time.perf_counter()
        chunks = retrieve(
            question,
            self._embedder,
            self._store,
            top_k=top_k if top_k is not None else self._top_k,
            min_score=min_score if min_score is not None else self._min_score,
        )
        logger.info("Retrieved %d chunks in %.2fs", len(chunks), time.perf_counter() - t0)
        user_prompt = build_user_prompt(chunks, question)
        answer = self._llm.generate(SYSTEM_PROMPT, user_prompt)
        logger.info("Generation done in %.2fs", time.perf_counter() - t0)

        citations: list[Citation] = []
        for index in extract_cited_indices(answer):
            if 1 <= index <= len(chunks):  # out-of-range [n] is dropped, never a crash
                chunk = chunks[index - 1].chunk
                citations.append(
                    Citation(
                        chunk_id=chunk.chunk_id,
                        source_file=chunk.source_file,
                        page_number=chunk.page_number,
                        text=chunk.text,
                    )
                )
        return AnswerResult(answer=answer, citations=citations, used_chunks=chunks)
