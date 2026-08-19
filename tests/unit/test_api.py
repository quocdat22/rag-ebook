"""Unit tests for src.api.main — TestClient with fake pipelines, no real services."""

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.schemas import QueryResponse
from src.pipeline.query_pipeline import AnswerResult, Citation
from src.vectorstore.chroma_store import SourceInfo

FIXTURE_PDF = Path("tests/fixtures/sample_tech_ebook.pdf")


class FakeIndexPipeline:
    def __init__(self, chunks_indexed: int = 3, error: Exception | None = None):
        self.chunks_indexed = chunks_indexed
        self.error = error
        self.calls: list[str] = []

    def run(self, pdf_path: str) -> int:
        self.calls.append(pdf_path)
        if self.error is not None:
            raise self.error
        return self.chunks_indexed


class FakeQueryPipeline:
    def __init__(self, result: AnswerResult | None = None, error: Exception | None = None):
        self.result = result or AnswerResult(
            answer="The answer is in the context [1].",
            citations=[
                Citation(chunk_id="c1", source_file="book.pdf", page_number=2, text="ctx text")
            ],
            used_chunks=[],
        )
        self.error = error
        self.calls: list[tuple[str, int | None, float | None]] = []

    def run(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> AnswerResult:
        self.calls.append((question, top_k, min_score))
        if self.error is not None:
            raise self.error
        return self.result


class FakeStore:
    """Minimal VectorStore fake for list/delete endpoints; records calls."""

    def __init__(self, sources: list[SourceInfo] | None = None):
        self.sources = sources or []
        self.delete_calls: list[str] = []
        self.delete_results: dict[str, int] = {}

    def list_sources(self) -> list[SourceInfo]:
        return self.sources

    def delete_by_source(self, source_file: str) -> int:
        self.delete_calls.append(source_file)
        return self.delete_results.get(source_file, 0)


def make_app(index=None, query=None, store=None, tmp_path: Path = Path("/tmp")):
    index = index if index is not None else FakeIndexPipeline()
    query = query if query is not None else FakeQueryPipeline()
    store = store if store is not None else FakeStore()
    return create_app(index, query, store, upload_dir=tmp_path)


def test_health():
    client = TestClient(make_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_schema(tmp_path):
    client = TestClient(make_app(tmp_path=tmp_path))
    response = client.post("/query", json={"question": "What is a vector database?"})
    assert response.status_code == 200
    body = response.json()
    parsed = QueryResponse.model_validate(body)  # shape matches the schema
    assert parsed.answer == body["answer"]
    assert parsed.citations[0].chunk_id == "c1"
    assert parsed.citations[0].page_number == 2


def test_query_empty_question(tmp_path):
    client = TestClient(make_app(tmp_path=tmp_path))
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_passes_per_request_overrides(tmp_path):
    query = FakeQueryPipeline()
    client = TestClient(make_app(query=query, tmp_path=tmp_path))
    response = client.post("/query", json={"question": "q", "top_k": 2, "min_score": 0.4})
    assert response.status_code == 200
    assert query.calls[-1] == ("q", 2, 0.4)


def test_query_defaults_to_none_overrides(tmp_path):
    query = FakeQueryPipeline()
    client = TestClient(make_app(query=query, tmp_path=tmp_path))
    response = client.post("/query", json={"question": "q"})
    assert response.status_code == 200
    # None -> QueryPipeline falls back to the values from Settings.
    assert query.calls[-1] == ("q", None, None)


def test_query_rejects_invalid_min_score(tmp_path):
    client = TestClient(make_app(tmp_path=tmp_path))
    response = client.post("/query", json={"question": "q", "min_score": 1.5})
    assert response.status_code == 422


def test_query_rejects_invalid_top_k(tmp_path):
    client = TestClient(make_app(tmp_path=tmp_path))
    response = client.post("/query", json={"question": "q", "top_k": 0})
    assert response.status_code == 422


def test_documents_uploads(tmp_path):
    index = FakeIndexPipeline(chunks_indexed=5)
    client = TestClient(make_app(index=index, tmp_path=tmp_path))
    with FIXTURE_PDF.open("rb") as pdf:
        response = client.post(
            "/documents",
            files={"file": ("sample_tech_ebook.pdf", pdf, "application/pdf")},
        )
    assert response.status_code == 200
    assert response.json() == {"filename": "sample_tech_ebook.pdf", "chunks_indexed": 5}
    assert index.calls  # the pipeline received a path to the saved upload


def test_documents_rejects_non_pdf(tmp_path):
    client = TestClient(make_app(tmp_path=tmp_path))
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", BytesIO(b"not a pdf"), "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_documents_list_empty(tmp_path):
    client = TestClient(make_app(tmp_path=tmp_path))
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == {"documents": []}


def test_documents_list_returns_sources(tmp_path):
    store = FakeStore(
        sources=[
            SourceInfo(source_file="alpha.pdf", chunk_count=1),
            SourceInfo(source_file="zeta.pdf", chunk_count=5),
        ]
    )
    client = TestClient(make_app(store=store, tmp_path=tmp_path))
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == {
        "documents": [
            {"filename": "alpha.pdf", "chunks": 1},
            {"filename": "zeta.pdf", "chunks": 5},
        ]
    }


def test_documents_delete_existing(tmp_path):
    store = FakeStore()
    store.delete_results["book.pdf"] = 12
    client = TestClient(make_app(store=store, tmp_path=tmp_path))
    response = client.delete("/documents/book.pdf")
    assert response.status_code == 200
    assert response.json() == {"filename": "book.pdf", "chunks_deleted": 12}
    assert store.delete_calls == ["book.pdf"]


def test_documents_delete_missing_404(tmp_path):
    store = FakeStore()
    client = TestClient(make_app(store=store, tmp_path=tmp_path))
    response = client.delete("/documents/never-indexed.pdf")
    assert response.status_code == 404
    assert "never-indexed.pdf" in response.json()["detail"]
    assert store.delete_calls == ["never-indexed.pdf"]


def test_pipeline_error_maps_to_500(tmp_path):
    index = FakeIndexPipeline()
    query = FakeQueryPipeline(error=RuntimeError("boom"))
    client = TestClient(make_app(index=index, query=query, tmp_path=tmp_path))
    response = client.post("/query", json={"question": "q"})
    assert response.status_code == 500
    assert "Traceback" not in response.text  # no raw traceback leak


def test_ollama_down_maps_to_503(tmp_path):
    from src.errors import OllamaUnavailableError

    query = FakeQueryPipeline(
        error=OllamaUnavailableError("Ollama not running — try: ollama serve")
    )
    client = TestClient(make_app(query=query, tmp_path=tmp_path))
    response = client.post("/query", json={"question": "q"})
    assert response.status_code == 503
    assert "ollama serve" in response.json()["detail"]


def test_empty_pdf_maps_to_400(tmp_path):
    from src.errors import EmptyDocumentError

    index = FakeIndexPipeline(
        error=EmptyDocumentError("No extractable text in PDF (scanned/image-only?)")
    )
    client = TestClient(make_app(index=index, tmp_path=tmp_path))
    with FIXTURE_PDF.open("rb") as pdf:
        response = client.post("/documents", files={"file": ("blank.pdf", pdf, "application/pdf")})
    assert response.status_code == 400
    assert "No extractable text" in response.json()["detail"]
