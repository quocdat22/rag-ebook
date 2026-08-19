"""FastAPI app for the RAG system (SPEC 5.8, F7).

Endpoints:
- ``GET /health``        -> {"status": "ok"} (no network probes — health checks
                           must not depend on Ollama/DeepSeek being up).
- ``POST /documents``    -> upload + index a PDF.
- ``POST /query``        -> question -> answer with citations.

``create_app`` takes the pipelines as dependencies (dependency injection) so
tests can pass fakes. ``app`` (module level) wires the real services from
``Settings`` for ``uvicorn src.api.main:app``.
"""

import logging
import re
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.api.schemas import CitationOut, IngestResponse, QueryRequest, QueryResponse
from src.config import Settings
from src.embedding.ollama_client import OllamaEmbeddingClient, OllamaUnavailableError
from src.generation.deepseek_client import DeepSeekClient, GenerationError
from src.ingestion.pdf_loader import EmptyDocumentError
from src.pipeline.index_pipeline import IndexPipeline
from src.pipeline.query_pipeline import QueryPipeline
from src.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/uploads")
DEFAULT_COLLECTION = "rag_ebook"  # all PDFs share one collection; filtered by source_file


def sanitize_filename(name: str) -> str:
    """Sanitize a client-supplied filename: basename only, safe characters only."""
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)
    return clean or "upload.pdf"


def create_app(
    index_pipeline: IndexPipeline,
    query_pipeline: QueryPipeline,
    upload_dir: Path = UPLOAD_DIR,
) -> FastAPI:
    app = FastAPI(title="rag-ebook", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/documents", response_model=IngestResponse)
    async def ingest_document(file: Annotated[UploadFile, File()]) -> IngestResponse:
        filename = sanitize_filename(file.filename or "upload.pdf")
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / filename
        dest.write_bytes(content)
        try:
            chunks_indexed = index_pipeline.run(str(dest))
        except (EmptyDocumentError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # never leak a raw traceback to the client
            logger.exception("Ingest failed for %s", filename)
            raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
        return IngestResponse(filename=filename, chunks_indexed=chunks_indexed)

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        try:
            result = query_pipeline.run(request.question)
        except GenerationError as exc:
            raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc
        except OllamaUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # never leak a raw traceback to the client
            logger.exception("Query failed for %r", request.question)
            raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
        return QueryResponse(
            answer=result.answer,
            citations=[CitationOut(**citation.model_dump()) for citation in result.citations],
        )

    return app


def _default_pipelines() -> tuple[IndexPipeline, QueryPipeline]:
    settings = Settings()
    store = ChromaVectorStore(
        collection_name=DEFAULT_COLLECTION, persist_dir=settings.chroma_persist_dir
    )
    embedder = OllamaEmbeddingClient(model=settings.ollama_embed_model, host=settings.ollama_host)
    llm = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        base_url=settings.deepseek_base_url,
    )
    index_pipeline = IndexPipeline(embedder, store, settings.chunk_size, settings.chunk_overlap)
    query_pipeline = QueryPipeline(embedder, store, llm, settings.top_k)
    return index_pipeline, query_pipeline


index_pipeline, query_pipeline = _default_pipelines()
app = create_app(index_pipeline, query_pipeline)
