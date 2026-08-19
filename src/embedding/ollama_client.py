"""Ollama embedding client (local, free) for document chunks and queries.

Uses Ollama's ``/api/embeddings`` endpoint (one prompt per request) with the
instruction-aware model ``qwen3-embedding:0.6b`` (1024-dim, context 32k):

- ``embed(texts)``  -> raw text, for documents/chunks (no prefix needed).
- ``embed_query()`` -> prepends ``Instruct: <query_instruction>\\nQuery: <text>``,
  which measurably improves retrieval quality for queries (SPEC 5.3).

Failures are loud, never silent (SPEC 5.3): connection/timeout errors raise
``OllamaUnavailableError`` with an actionable hint; HTTP errors raise
``OllamaHTTPError`` (404 hints at a missing model); unexpected/empty embeddings
or a wrong dimension raise ``ValueError``.

Optimization note: Ollama also offers ``/api/embed`` accepting a batch payload
``{"input": [...]}``. ``embed(texts)`` is already batch-shaped, so switching
implementations later does not change the interface.
"""

from typing import Protocol, Self

import httpx

DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class OllamaUnavailableError(RuntimeError):
    """Ollama is not running, unreachable, or did not respond in time."""


class OllamaHTTPError(RuntimeError):
    """Ollama answered with a non-2xx HTTP status."""


class EmbeddingClient(Protocol):
    """Embedding provider contract (SPEC 5.3) — swap-in-able / mockable."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbeddingClient:
    """Embedding client for a local Ollama server."""

    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        host: str = "http://localhost:11434",
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        timeout: float = 60.0,
        dim: int = 1024,
    ) -> None:
        self._model = model
        self._url = f"{host.rstrip('/')}/api/embeddings"
        self._query_instruction = query_instruction
        self._timeout = timeout
        self._dim = dim
        # Reused connection; close via close() or the context manager.
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed document/chunk texts (raw, no instruction prefix)."""
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the instruction prefix for best retrieval quality."""
        prompt = f"Instruct: {self._query_instruction}\nQuery: {text}"
        return self._embed_one(prompt)

    def _embed_one(self, prompt: str) -> list[float]:
        try:
            response = self._client.post(self._url, json={"model": self._model, "prompt": prompt})
            response.raise_for_status()
        except httpx.RequestError as exc:
            if isinstance(exc, httpx.TimeoutException):
                detail = f"timed out after {self._timeout}s"
            else:
                detail = "could not connect"
            raise OllamaUnavailableError(
                f"Ollama embedding failed ({detail}). Is Ollama running? Try: ollama serve "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        except httpx.HTTPStatusError as exc:
            hint = (
                f" (model missing? try: ollama pull {self._model})"
                if exc.response.status_code == 404
                else ""
            )
            raise OllamaHTTPError(
                f"Ollama returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:300]!r}{hint}"
            ) from exc

        try:
            embedding = response.json().get("embedding")
        except ValueError as exc:  # invalid JSON body
            raise ValueError(
                f"Unexpected Ollama response (invalid JSON): {response.text[:200]!r}"
            ) from exc
        if not embedding:
            raise ValueError(f"Ollama returned an empty embedding for prompt: {prompt[:80]!r}")
        if len(embedding) != self._dim:
            raise ValueError(
                f"Embedding dimension {len(embedding)} != expected {self._dim}. "
                f"Model changed? Changing the embedding model requires a full re-index "
                f"(dimension mismatch breaks existing collections)."
            )
        return [float(value) for value in embedding]
