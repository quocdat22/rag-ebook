"""API request/response schemas (SPEC 5.8)."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)  # None -> use settings
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)  # None -> use settings


class CitationOut(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
