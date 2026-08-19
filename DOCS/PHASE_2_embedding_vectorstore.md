# PHASE 2 — Embedding (Ollama) + Vector store (Chroma)

**Trạng thái:** [x] Hoàn thành
**Phụ thuộc:** Phase 1 (dùng `Chunk`)
**Output:** Index được 1 PDF vào Chroma; cả 2 module có unit test với mock, không cần service thật.

---

## 1. Mục tiêu

- `OllamaEmbeddingClient`: gọi Ollama local, sinh vector 1024-dim (model `qwen3-embedding:0.6b`), lỗi rõ ràng khi Ollama không chạy (SPEC 5.3).
- `ChromaVectorStore`: lưu chunk + embedding vào Chroma persist local; query theo cosine similarity (SPEC 5.4).
- Cả 2 module triển khai qua `Protocol` → thay được provider khác và test bằng mock dễ dàng (SPEC mục 3, 6).

## 2. Thiết kế

### 2.1 `src/embedding/ollama_client.py`

```python
from typing import Protocol

import httpx


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OllamaUnavailableError(RuntimeError):
    """Ollama không chạy hoặc không phản hồi."""


class OllamaEmbeddingClient:
    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        host: str = "http://localhost:11434",
        query_instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
        timeout: float = 60.0,
    ): ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

Implementation notes:
- **`qwen3-embedding:0.6b` là model instruction-aware** (1024-dim, context 32k): phía query phải có prefix `Instruct: <query_instruction>\nQuery: <text>` để đạt chất lượng tốt nhất (+1–5%); phía document/chunk **không** cần prefix. Do đó tách 2 hàm:
  - `embed(texts)` → dùng cho chunk/document, gửi text thô.
  - `embed_query(text)` → prepend instruction rồi mới gửi (dùng trong Phase 3 khi embed câu hỏi).
- Gọi `POST {host}/api/embeddings` với payload `{"model": model, "prompt": text}` cho từng text (API `/api/embeddings` nhận 1 prompt/lần).
- Response JSON: `{"embedding": [1024 floats]}`.
- **Xử lý lỗi (không crash im lặng — SPEC 5.3):**
  - `httpx.ConnectError` / timeout → raise `OllamaUnavailableError` kèm message gợi ý hành động: *"Ollama chưa chạy? Thử: ollama serve"*.
  - Status ≠ 200 → raise kèm status code + body; 404 → gợi ý `ollama pull qwen3-embedding:0.6b`.
  - Kết quả rỗng hoặc dimension ≠ 1024 → raise `ValueError` rõ ràng.
- Dùng `httpx.Client` tái sử dụng connection (đóng qua context manager hoặc `close()`).

> **Tối ưu sau (ghi chú trong code):** Ollama có `/api/embed` nhận batch `{"input": [...]}` — interface `embed(texts)` đã thiết kế sẵn cho batch, chỉ cần đổi implementation sau, không đổi interface.

### 2.2 `src/vectorstore/chroma_store.py`

```python
from typing import Protocol

from pydantic import BaseModel

from src.chunking.splitter import Chunk


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float  # cosine similarity: càng cao càng giống


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def query(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]: ...
    def delete_by_source(self, source_file: str) -> int: ...
    def list_sources(self) -> list[SourceInfo]: ...


class ChromaVectorStore:
    def __init__(self, collection_name: str, persist_dir: str): ...
```

Implementation notes:
- `chromadb.PersistentClient(path=persist_dir)`; `get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})` — **cosine là chuẩn cho RAG text**.
- `add()`: `collection.add(ids=[c.chunk_id...], documents=[c.text...], embeddings=embeddings, metadatas=[{"page_number": ..., "source_file": ...}])`; validate số lượng khớp nhau trước khi add.
- `query()`: `collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances"])` → build lại `RetrievedChunk`; `score = 1 - distance` (cosine distance → similarity).
- `delete_by_source(source_file)`: `collection.get(where={"source_file": ...})` → `collection.delete(ids)`; trả số chunk đã xoá. Dùng cho re-index (xoá cũ trước khi add mới) và API `DELETE /documents/{file}`.
- `list_sources()`: `collection.get(include=["metadatas"])` → aggregate đếm theo `source_file`, sorted theo filename; trả `list[SourceInfo]` (phục vụ API `GET /documents`). Với collection rất lớn nên dùng phân trang — chấp nhận toàn bộ ở MVP.
- **Test chạy in-memory:** constructor nhận `persist_dir` — test dùng `chromadb.EphemeralClient()` (persist_dir=None) hoặc `tempfile.mkdtemp()`.

## 3. Test chi tiết

### `tests/unit/test_ollama_client.py` — mock HTTP, **không gọi Ollama thật**

Mock `httpx.Client.post` (pytest-mock):

| Test | Assert |
|---|---|
| `test_payload_correct` | gọi đúng URL `{host}/api/embeddings`, payload `{"model": "qwen3-embedding:0.6b", "prompt": "..."}` |
| `test_returns_1024_dim` | mock response `{"embedding": [0.1] * 1024}` → `len(result[0]) == 1024` |
| `test_embed_query_adds_instruction` | `embed_query("...")` gửi prompt có prefix `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ...` |
| `test_connect_error_raises` | mock raise `httpx.ConnectError` → `pytest.raises(OllamaUnavailableError)` |
| `test_timeout_raises` | mock raise `httpx.ReadTimeout` → `OllamaUnavailableError` |
| `test_http_error_raises` | mock 404 → exception chứa gợi ý pull model |
| `test_wrong_dimension_raises` | mock trả vector 128-dim → `ValueError` |
| `test_batch_multiple_texts` | 2 texts → 2 lần gọi HTTP, trả 2 vector |

### `tests/unit/test_chroma_store.py` — Chroma thật (in-memory/temp dir), không cần mock

Dùng vector tay đơn giản để kiểm chứng similarity bằng mắt:

| Test | Assert |
|---|---|
| `test_add_and_query_top1` | add chunks với vector `[1,0,0]`, `[0,1,0]`, `[1,1,0]`; query `[0.9,0.1,0]` → top-1 là chunk đầu |
| `test_top_k_limits` | 5 chunks, query `top_k=2` → trả đúng 2 |
| `test_metadata_roundtrip` | `RetrievedChunk.chunk.page_number` / `source_file` đúng bản ghi gốc |
| `test_score_range` | mọi `0 <= score <= 1`; chunk trùng vector query → score ≈ 1.0 |
| `test_empty_store_returns_empty` | collection rỗng → `[]` |
| `test_add_duplicate_ids_defined_behavior` | add 2 lần cùng id → hành vi xác định (skip hoặc error rõ ràng, không crash bí hiểm) |

## 4. Kiểm tra thủ công (smoke, dùng Ollama thật)

```bash
# Script tạm: load_pdf → split → embed (Ollama thật) → Chroma
uv run python scripts/smoke_index.py
# Kỳ vọng: in số chunk đã index; thư mục data/chroma/ xuất hiện
```

## 5. Tiêu chí hoàn thành (DoD)

- [x] Unit test 2 module pass — **không cần Ollama chạy**
- [x] Smoke index fixture PDF vào `data/chroma` thành công với Ollama thật
- [x] `uv run ruff check --fix . && uv run ruff format . && uv run pytest` → pass
- [x] `data/` không bị commit (gitignore hoạt động)

## 6. Rủi ro & lưu ý

- **Dimension mismatch** (đổi model embedding) phá collection cũ → luôn validate 1024-dim trước khi add; đổi model = phải re-index toàn bộ (ghi chú trong code).
- **`/api/embeddings` 1 prompt/lần** → nhiều chunk = nhiều request; batch với `/api/embed` để tối ưu ở Phase 7 nếu còn thời gian.
- **Chroma API đổi giữa các version** — pin version trong `uv.lock`, chạy lại test ngay khi nâng cấp.
- **`score = 1 - distance` chỉ đúng với cosine** — đã cố định `hnsw:space=cosine`, không đổi sang L2/IP mà quên sửa chỗ này.
