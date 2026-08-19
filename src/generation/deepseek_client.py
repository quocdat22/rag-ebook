"""DeepSeek LLM client via the OpenAI-compatible SDK (SPEC 5.6, F5).

Uses the modern model names directly — ``deepseek-v4-flash`` (default) /
``deepseek-v4-pro``. The deprecated aliases ``deepseek-chat`` /
``deepseek-reasoner`` are gone and must not be used.
"""

from typing import Protocol

import openai
from openai import OpenAI

# Errors considered "transient/expected" for a generation call; everything else
# (e.g. programming mistakes) propagates as-is.
_TRANSIENT_ERRORS = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
)


class LLMClient(Protocol):
    """LLM provider contract (SPEC 5.6) — swap-in-able / mockable."""

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class GenerationError(RuntimeError):
    """Raised when DeepSeek fails or returns an unusable response."""


class DeepSeekClient:
    """DeepSeek API client (OpenAI-compatible endpoint)."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout: float = 60.0,
        client: OpenAI | None = None,  # inject a fake in tests
    ) -> None:
        self._model = model
        self._client = (
            client
            if client is not None
            else OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call the chat endpoint and return the assistant's text content.

        Low temperature (0.2): RAG answers must stick to the context, not be
        creative. Empty/missing content or transient API errors raise
        ``GenerationError`` — never a silent empty string.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
        except _TRANSIENT_ERRORS as exc:
            raise GenerationError(
                f"DeepSeek generation failed ({type(exc).__name__}): {exc}"
            ) from exc
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError):
            content = None
        if not content:
            raise GenerationError("DeepSeek returned an empty response (no content)")
        return content
