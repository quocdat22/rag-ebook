# PHASE 5 — Pipeline + FastAPI

**Trạng thái:** [x] Hoàn thành
**Phụ thuộc:** Phase 3 + Phase 4
**Output:** API `/query` trả lời đúng kèm citation; integration test end-to-end chạy được **không cần** Ollama/DeepSeek thật.

---

## 1. Mục tiêu

- 2 pipeline: `index_pipeline` (ingest PDF → Chroma) và `query_pipeline` (question → `AnswerResult`) (SPEC 5.7).
- FastAPI: `POST /documents`, `POST /query`, `GET /health` (SPEC 5.8, F7).
- Integration test full pipeline với fake embedder + fake LLM + Chroma thật (temp dir) (SPEC mục 6).

## 2. Thiết kế

### 2.1 Model kết quả — `src/pipeline/query_pipeline.py`

```python
class Citation(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int
    text: str


class AnswerResult(BaseModel):
    answer: str
    citations: list[Citation]  # các nguồn thực sự được trích [n]
    used_chunks: list[RetrievedChunk]  # toàn bộ context đã đưa vào prompt
```

### 2.2 `src/pipeline/index_pipeline.py`

```python
class IndexPipeline:
    def __init__(
        self,
        embedder: EmbeddingClient,
        store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
    ): ...
    def run(self, pdf_path: str) -> int: ...  # trả về số chunk đã index
```

Luồng: `load_pdf → split_documents(chunk_size, chunk_overlap) → embedder.embed([c.text ...]) → store.add`.

- **Idempotency:** index lại cùng 1 file → xoá chunk cũ trước (`collection.delete(where={"source_file": ...})`) để không nhân bản dữ liệu.
- PDF rỗng (0 chunk) → raise rõ ràng, **không** ghi gì vào store.
- Log các mốc chính: số trang → số chunk → thời gian embed (hoàn thiện logging ở Phase 7).

### 2.3 `src/pipeline/query_pipeline.py`

```python
class QueryPipeline:
    def __init__(
        self,
        embedder: EmbeddingClient,
        store: VectorStore,
        llm: LLMClient,
        top_k: int,
        min_score: float,
    ): ...
    def run(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> AnswerResult: ...  # None → dùng giá trị từ constructor (Settings)
```

Luồng (bám SPEC 4.2): `retrieve(question, ...) → build_user_prompt(chunks, question) → llm.generate(SYSTEM_PROMPT, user_prompt) → extract_cited_indices(answer) → map index → Citation`.

- Index trích dẫn ngoài range `[1..len(chunks)]` → **bỏ qua, không crash** (model đôi khi sinh `[99]`).
- `used_chunks` giữ toàn bộ context (để debug/báo cáo), `citations` chỉ chứa nguồn thực sự được trích.

### 2.4 `src/api/schemas.py` + `src/api/main.py`

```python
# schemas.py
class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None  # None → lấy từ settings
    min_score: float | None = None  # None → lấy từ settings; ngoài [0, 1] → 422


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
```

Endpoints:

| Endpoint | Hành vi |
|---|---|
| `POST /documents` | nhận `UploadFile` PDF → lưu tạm `data/uploads/` → `index_pipeline.run(...)` → `IngestResponse`. File không phải PDF / rỗng → 400/422 với message rõ ràng |
| `POST /query` | `QueryRequest` (hỗ trợ override `top_k`/`min_score` per-request, `None` → dùng settings) → `QueryResponse` |
| `GET /health` | `{"status": "ok"}` (probe Ollama/DeepSeek để **false** mặc định — không bắt health-check phải gọi network) |

App factory để test dễ mock:

```python
def create_app(index_pipeline: IndexPipeline, query_pipeline: QueryPipeline) -> FastAPI: ...
```

Chạy: `uv run uvicorn src.api.main:app --reload` (main.py tự build app mặc định từ `settings`).

> `UploadFile` cần `python-multipart` — đã cài từ Phase 0. Nếu quên: `uv add python-multipart`.

## 3. Test chi tiết

### `tests/integration/test_end_to_end_pipeline.py` — đánh dấu `@pytest.mark.integration`

Dùng **PDF fixture thật + Chroma thật (temp dir)** nhưng **fake embedder + fake LLM** → chạy được mọi nơi, kể cả CI (SPEC mục 6):

- `test_index_then_query_full_flow`:
  1. FakeEmbedder trả vector giả định (vd: hash đơn giản theo text).
  2. `IndexPipeline.run(fixture_pdf)` → số chunk > 0.
  3. `QueryPipeline.run("What is ...?")` → `AnswerResult`.
  4. Assert: `answer` không rỗng; `used_chunks` không rỗng; mọi `citations` có `chunk_id` nằm trong `used_chunks`; mọi Citation đủ field; `page_number` là int hợp lệ.
- `test_reindex_same_file_no_duplicates`: run 2 lần → tổng chunk trong collection không tăng gấp đôi.

### `tests/unit/test_api.py` — `fastapi.testclient.TestClient`

Tạo app bằng `create_app()` với **pipeline fake** (không cần Chroma/Ollama):

| Test | Assert |
|---|---|
| `test_health` | GET /health → 200, body `{"status": "ok"}` |
| `test_query_returns_schema` | POST /query → 200, response validate đúng `QueryResponse` |
| `test_query_empty_question` | question rỗng → 422 |
| `test_documents_uploads` | POST /documents với PDF fixture (multipart) → 200, `chunks_indexed > 0` |
| `test_documents_rejects_non_pdf` | upload file .txt → 4xx |
| `test_pipeline_error_maps_to_500` | fake pipeline raise → 500 (hoặc mã đã chọn), không trả traceback trần |

### `tests/unit/test_pipelines.py` (bổ sung, nếu integration chưa đủ chi tiết)

- query_pipeline: fake retrieve trả 2 chunks; fake LLM trả `"answer [1] [99]"` → `citations` chỉ chứa index 1 (99 bị bỏ).
- index_pipeline: PDF 0 trang → raise, store không bị gọi `add`.

## 4. Kiểm tra thủ công (lần đầu dùng service thật)

```bash
uv run uvicorn src.api.main:app --reload

# health
curl http://127.0.0.1:8000/health

# ingest (cần Ollama chạy)
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@tests/fixtures/sample_tech_ebook.pdf"

# query (cần DEEPSEEK_API_KEY trong .env)
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a vector database?"}'
# Kỳ vọng: answer có marker [n] + citations list có file/page
```

Hoặc mở http://127.0.0.1:8000/docs (Swagger UI) để thử trực quan.

## 5. Tiêu chí hoàn thành (DoD)

- [x] Integration + API test pass — **không cần** Ollama/DeepSeek chạy
- [x] Chạy tay cả 3 endpoint bằng service thật → trả lời đúng kèm citation đúng trang
- [x] `uv run ruff check --fix . && uv run ruff format . && uv run pytest` → pass (integration test chạy đầy đủ ở local)

## 6. Rủi ro & lưu ý

- **Ingest chạy đồng bộ trong request** → PDF lớn làm request lâu; MVP chấp nhận (ghi chú). V2: background task + status polling (SPEC mục 10).
- **Filename từ client không tin cậy** — sanitize trước khi lưu vào `data/uploads/` (chống path traversal).
- **Câu hỏi ngoài phạm vi tài liệu:** system prompt yêu cầu model nói rõ "context không chứa đáp án" — thử vài câu lạc đề khi demo để kiểm chứng.
- **Nhiều PDF:** mọi file chung collection `rag_ebook`, lọc bằng metadata `source_file`; query trên tất cả (câu trả lời có thể dẫn nhiều sách) — phù hợp MVP "1 hoặc nhiều PDF" (SPEC mục 1).
