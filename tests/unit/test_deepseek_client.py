"""Unit tests for src.generation.deepseek_client — fake OpenAI client, no real API."""

import httpx2
import openai
import pytest

from src.generation.deepseek_client import DeepSeekClient, GenerationError
from src.generation.prompt_templates import SYSTEM_PROMPT

ENDPOINT = "https://api.deepseek.com/chat/completions"


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeOpenAI:
    def __init__(self, completions):
        self.chat = FakeChat(completions)


def make_client(completions: FakeCompletions, model: str = "deepseek-v4-flash"):
    fake = FakeOpenAI(completions)
    client = DeepSeekClient(api_key="test-key", model=model, client=fake)
    return client, completions


def test_generate_returns_content():
    client, _ = make_client(FakeCompletions(FakeResponse("The answer is 42.")))
    assert client.generate("sys", "user") == "The answer is 42."


def test_messages_format():
    client, completions = make_client(FakeCompletions(FakeResponse("ok")))
    client.generate(SYSTEM_PROMPT, "user question")
    kwargs = completions.calls[0]
    assert kwargs["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "user question"},
    ]


def test_model_and_temperature_passed():
    client, completions = make_client(
        FakeCompletions(FakeResponse("ok")), model="deepseek-v4-flash"
    )
    client.generate("s", "u")
    kwargs = completions.calls[0]
    assert kwargs["model"] == "deepseek-v4-flash"
    assert kwargs["temperature"] == 0.2


def test_empty_response_raises():
    client, _ = make_client(FakeCompletions(FakeResponse(None)))
    with pytest.raises(GenerationError, match="empty"):
        client.generate("s", "u")


def test_timeout_raises():
    request = httpx2.Request("POST", ENDPOINT)
    error = openai.APITimeoutError(request)
    client, _ = make_client(FakeCompletions(error=error))
    with pytest.raises(GenerationError, match="timed out"):
        client.generate("s", "u")


def test_connection_error_raises():
    request = httpx2.Request("POST", ENDPOINT)
    error = openai.APIConnectionError(message="connection failed", request=request)
    client, _ = make_client(FakeCompletions(error=error))
    with pytest.raises(GenerationError):
        client.generate("s", "u")


def test_rate_limit_raises():
    request = httpx2.Request("POST", ENDPOINT)
    error = openai.RateLimitError(
        "rate limited", response=httpx2.Response(429, request=request), body=None
    )
    client, _ = make_client(FakeCompletions(error=error))
    with pytest.raises(GenerationError, match="rate|429"):
        client.generate("s", "u")
