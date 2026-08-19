"""Unit tests for src.embedding.ollama_client — HTTP mocked, no real Ollama."""

import httpx
import pytest

from src.embedding.ollama_client import (
    OllamaEmbeddingClient,
    OllamaHTTPError,
    OllamaUnavailableError,
)

HOST = "http://ollama.test:11434"
MODEL = "qwen3-embedding:0.6b"
URL = f"{HOST}/api/embeddings"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def make_client(**kwargs) -> OllamaEmbeddingClient:
    return OllamaEmbeddingClient(model=MODEL, host=HOST, **kwargs)


def mock_ok_post(mocker, dim: int = 1024):
    """Patch httpx.Client.post to return a 200 response with a `dim`-dim vector."""
    mock_post = mocker.patch("httpx.Client.post")
    mock_post.return_value.status_code = 200
    mock_post.return_value.text = '{"embedding": [0.1]}'
    mock_post.return_value.json.return_value = {"embedding": [0.1] * dim}
    return mock_post


def test_payload_correct(mocker):
    mock_post = mock_ok_post(mocker)
    client = make_client()
    client.embed(["hello world"])
    assert mock_post.call_count == 1
    call = mock_post.call_args
    assert call.args[0] == URL
    assert call.kwargs["json"] == {"model": MODEL, "prompt": "hello world"}


def test_returns_1024_dim(mocker):
    mock_ok_post(mocker)
    client = make_client()
    result = client.embed(["some text"])
    assert len(result) == 1
    assert len(result[0]) == 1024


def test_embed_query_adds_instruction(mocker):
    mock_post = mock_ok_post(mocker)
    client = make_client()
    client.embed_query("what is overfitting?")
    prompt = mock_post.call_args.kwargs["json"]["prompt"]
    assert prompt == f"Instruct: {INSTRUCTION}\nQuery: what is overfitting?"


def test_connect_error_raises(mocker):
    mocker.patch("httpx.Client.post", side_effect=httpx.ConnectError("connection refused"))
    client = make_client()
    with pytest.raises(OllamaUnavailableError, match="ollama serve"):
        client.embed(["text"])


def test_timeout_raises(mocker):
    mocker.patch("httpx.Client.post", side_effect=httpx.ReadTimeout("slow response"))
    client = make_client()
    with pytest.raises(OllamaUnavailableError):
        client.embed(["text"])


def test_http_error_raises_with_pull_hint(mocker):
    request = httpx.Request("POST", URL)
    mocker.patch("httpx.Client.post", return_value=httpx.Response(404, request=request))
    client = make_client()
    with pytest.raises(OllamaHTTPError, match=f"ollama pull {MODEL}"):
        client.embed(["text"])


def test_http_500_raises_with_status(mocker):
    request = httpx.Request("POST", URL)
    mocker.patch("httpx.Client.post", return_value=httpx.Response(500, request=request))
    client = make_client()
    with pytest.raises(OllamaHTTPError, match="500"):
        client.embed(["text"])


def test_wrong_dimension_raises(mocker):
    mock_ok_post(mocker, dim=128)
    client = make_client()
    with pytest.raises(ValueError, match="1024"):
        client.embed(["text"])


def test_empty_embedding_raises(mocker):
    mock_ok_post(mocker, dim=0)
    client = make_client()
    with pytest.raises(ValueError, match="empty"):
        client.embed(["text"])


def test_batch_multiple_texts(mocker):
    mock_post = mock_ok_post(mocker)
    client = make_client()
    result = client.embed(["a", "b"])
    assert mock_post.call_count == 2
    assert len(result) == 2
    assert all(len(vector) == 1024 for vector in result)
