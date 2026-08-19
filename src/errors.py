"""Central exception hierarchy for rag-ebook (Phase 7).

All system-wide errors live here instead of being defined ad-hoc in each
module; modules import from this single place. Every message is *actionable* —
it tells the operator what to do (start Ollama, set the API key, ...), not
just what went wrong.
"""


class RagEbookError(Exception):
    """Base class for all rag-ebook errors."""


class EmptyDocumentError(RagEbookError):
    """PDF has no extractable text layer (empty or scanned/image-only)."""


class OllamaUnavailableError(RagEbookError):
    """Ollama is not running, unreachable, or did not respond in time."""


class OllamaHTTPError(RagEbookError):
    """Ollama answered with a non-2xx HTTP status."""


class GenerationError(RagEbookError):
    """LLM generation failed (timeout, rate limit, empty response...)."""


class ConfigurationError(RagEbookError):
    """Missing or invalid configuration (e.g. DEEPSEEK_API_KEY not set)."""
