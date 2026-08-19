# Kiến trúc rag-ebook — bản mô tả để vẽ SVG

## 1. Tổng quan

Hệ thống RAG (Retrieval-Augmented Generation) cho ebook PDF kỹ thuật tiếng Anh,
chạy **local, Python 3.12, không container**. Có **2 luồng chính**:

1. **Luồng Indexing (offline)**: PDF → text theo trang → chunk → embedding → vector store.
2. **Luồng Query (online)**: câu hỏi → embedding câu hỏi → top-k chunk → prompt → LLM → trả lời + trích dẫn `[n]`.

Ba dịch vụ ngoài:
- **Ollama** (local, miễn phí) — model embedding `qwen3-embedding:0.6b` (1024-dim, instruction-aware).
- **ChromaDB** (local persist, thư mục `./data/chroma`) — vector store, collection `rag_ebook`, không gian cosine.
- **DeepSeek API** (trả phí) — LLM `deepseek-v4-flash`, endpoint OpenAI-compatible `https://api.deepseek.com`.

Mọi module nhận dependency ngoài qua **constructor/tham số (DI)** với các Protocol
(`EmbeddingClient`, `VectorStore`, `LLMClient`) → test được bằng fake/mock, thay provider dễ dàng.

## 2. Sơ đồ lớp (layer)

```
┌─────────────────────────────────────────────────────────────┐
│  UI layer       src/ui/streamlit_app.py   (demo, gọi thẳng pipeline) │
│  API layer      src/api/main.py + schemas.py (FastAPI, DI qua create_app) │
├─────────────────────────────────────────────────────────────┤
│  Pipeline       src/pipeline/index_pipeline.py / query_pipeline.py │
├─────────────────────────────────────────────────────────────┤
│  Domain layer   ingestion → chunking → embedding → vectorstore → retrieval → generation │
├─────────────────────────────────────────────────────────────┤
│  Cross-cutting  src/config.py (.env) · src/errors.py · src/logging_config.py │
└─────────────────────────────────────────────────────────────┘
```

## 3. Luồng Indexing (chi tiết)

```
PDF ──▶ [1. ingestion] ──▶ Document[] ──▶ [2. chunking] ──▶ Chunk[]
        pdf_loader.py                    splitter.py
                                                              │
                    [4. vectorstore] ◀── embeddings ── [3. embedding]
                    chroma_store.py                          ollama_client.py
                    (Chroma persist,                        (Ollama /api/embeddings)
                     cosine, idempotent)
```

Các bước:
1. **Ingestion** — `load_pdf(path)` dùng **PyMuPDF** đọc text từng trang có nội dung.
   Kết quả: `Document {text, page_number (1-based), source_file (chỉ tên file)}`.
   Lỗi: file không phải PDF → `ValueError`; PDF scan không có text → `EmptyDocumentError`.
2. **Chunking** — `split_documents()` (splitter tự viết, **không dùng LangChain**):
   tách text trang thành "segment nguyên tử" (fenced code block ```…``` giữ nguyên vẹn,
   dòng heading, đoạn văn), rồi gói tham lam vào chunk ≤ ~700 token (ước lượng = số từ × 1.3),
   overlap ~100 token (đuôi chunk trước được mang sang chunk sau). Segment quá dài (code block lớn)
   được cắt theo dòng mới.
   Kết quả: `Chunk {chunk_id, text, page_number, source_file}` với
   `chunk_id = "{source_file}::p{page_number}::{index}"`.
3. **Embedding** — `embedder.embed([text...])`: gọi Ollama `/api/embeddings` (1 prompt/request),
   model `qwen3-embedding:0.6b`, vector 1024 chiều. Chunk dùng text thô (không prefix).
   Lỗi connection/timeout → `OllamaUnavailableError`; HTTP lỗi → `OllamaHTTPError`;
   sai số chiều → `ValueError` (đổi model embedding bắt buộc re-index toàn bộ).
4. **Vector store** — `store.add(chunks, embeddings)` vào ChromaDB persistent,
   collection `rag_ebook`, metadata mỗi chunk: `{page_number, source_file}`, không gian `hnsw:space: cosine`.
   **Idempotent**: trước khi add, xóa hết chunk cũ của cùng `source_file`
   (`delete_by_source`) — và chỉ xóa **sau khi** embedding mới thành công (fail giữa chừng
   giữ nguyên index cũ). Không nhân bản chunk; trùng `chunk_id` là lỗi.

## 4. Luồng Query (chi tiết)

```
Question ──▶ [1. retrieval] ──▶ top-k RetrievedChunk[] ──▶ [2. prompt builder]
              retriever.py                                 prompt_templates.py
   │                                                              │
   └─ embed_query (Ollama, prefix "Instruct: …")          [3. generation]
        └─ Chroma query (cosine)                          deepseek_client.py
        └─ lọc min_score (mặc định 0)                     (DeepSeek chat completions)
        └─ sort score giảm dần                                    │
                                                     [4. citation mapping]
                                                     answer + citations[] ──▶ Response
```

Các bước:
1. **Retrieval** — `retrieve(question, embedder, store, top_k=5, min_score=0.0)`:
   - `embed_query(question)` — **thêm prefix** `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: …"` (quan trọng cho model instruction-aware; không dùng `embed()` thô cho câu hỏi).
   - `store.query(embedding, top_k)` — Chroma cosine similarity, score = `1 − distance`.
   - Nếu `min_score > 0`: giữ chunk có `score ≥ min_score` (chống context không liên quan).
   - Sort theo score giảm dần. Store rỗng → trả về `[]`, không lỗi.
   Kết quả: `RetrievedChunk {chunk: Chunk, score: float}`.
2. **Prompt builder** — `build_user_prompt(chunks, question)` tạo user prompt:
   ```text
   Context:
   [1] (sample_tech_ebook.pdf, page 2)
   <text chunk 1>

   [2] (sample_tech_ebook.pdf, page 3)
   <text chunk 2>

   Question: …
   Answer with inline citations like [1].
   ```
   System prompt là **hằng số cố định** (`SYSTEM_PROMPT`): trả lời bám sát context,
   trích dẫn `[n]`, không bịa nguồn — cố định để tận dụng **prefix cache** của DeepSeek (rẻ hơn).
3. **Generation** — `llm.generate(SYSTEM_PROMPT, user_prompt)` qua OpenAI SDK,
   model `deepseek-v4-flash`, **temperature 0.2** (bám context, không sáng tạo).
   Lỗi timeout/rate-limit/connection → `GenerationError`; response rỗng → `GenerationError`
   (không bao giờ trả chuỗi rỗng im lặng).
4. **Citation mapping** — `extract_cited_indices(answer)` parse `[n]` theo thứ tự xuất hiện,
   dedupe; chỉ giữ `1 ≤ n ≤ len(chunks)` (ngoài khoảng bị bỏ, không crash), map ngược về chunk
   → `Citation {chunk_id, source_file, page_number, text}`.
   Kết quả cuối: `AnswerResult {answer, citations[], used_chunks[]}`.

## 5. Interface layer

### FastAPI — `src/api/main.py`
- `create_app(index_pipeline, query_pipeline, upload_dir)` — **factory + DI** (test truyền fake).
- Endpoints:
  - `GET /health` → `{"status": "ok"}` (không probe network — không phụ thuộc Ollama/DeepSeek).
  - `POST /documents` (multipart `file`) → sanitize tên file, lưu vào `data/uploads/`, chạy index
    pipeline đồng bộ → `{"filename", "chunks_indexed"}`. Chỉ nhận `.pdf`, file rỗng bị chặn.
  - `POST /query` (JSON `{"question", "top_k?"}`) → `{"answer", "citations": [{chunk_id, source_file, page_number, text}]}`.
- Exception handler tập trung cho hệ `RagEbookError`: `EmptyDocumentError`→400,
  `GenerationError`→502, còn lại→503; không bao giờ lộ traceback thô.
- `app` module-level wire service thật từ `Settings` (chạy `uvicorn src.api.main:app`).

### Streamlit — `src/ui/streamlit_app.py`
- Demo local (không auth, không deploy public), 2 tab: **Ingest PDF** (upload + index,
  hiện số chunk) và **Ask a question** (chat, hiện answer + expander Sources kèm chunk_id,
  file, trang, preview text).
- Gọi **thẳng pipeline layer trong cùng process** (không đi qua FastAPI).

## 6. Cross-cutting

- **Config** — `src/config.py`: pydantic-settings đọc `.env`:
  `OLLAMA_HOST` (localhost:11434), `OLLAMA_EMBED_MODEL` (qwen3-embedding:0.6b),
  `DEEPSEEK_API_KEY` (bắt buộc), `DEEPSEEK_MODEL` (deepseek-v4-flash),
  `DEEPSEEK_BASE_URL` (api.deepseek.com), `CHROMA_PERSIST_DIR` (./data/chroma),
  `CHUNK_SIZE` (700), `CHUNK_OVERLAP` (100), `TOP_K` (5), `LOG_LEVEL` (INFO).
- **Errors** — `src/errors.py`: `RagEbookError` (base) → `EmptyDocumentError`,
  `OllamaUnavailableError`, `OllamaHTTPError`, `GenerationError`, `ConfigurationError`.
  Mọi message đều kèm hướng xử lý (start Ollama, set API key…).
- **Logging** — `src/logging_config.py`: `setup_logging()` theo `LOG_LEVEL`.

## 7. Data models (luồng dữ liệu)

```
Document  {text, page_number, source_file}            (ingestion)
    │ split
Chunk     {chunk_id, text, page_number, source_file}  (chunking)  → lưu Chroma + metadata
    │ embed (1024-dim)
RetrievedChunk {chunk, score}                         (retrieval)
    │ prompt
AnswerResult {answer, citations[], used_chunks[]}     (query pipeline)
    │ map [n]
Citation  {chunk_id, source_file, page_number, text}  (API response)
```

## 8. Gợi ý bố cục SVG

- **Hai làn (swimlane) song song** — trái: "Indexing pipeline" (đi xuống), phải: "Query pipeline" (đi xuống), chia đôi màn hình.
- **Giữa hai làn**: 1 hộp chung **ChromaDB (vector store, persist)** — indexing ghi vào, query đọc ra.
- **Phía dưới/ngoài**: 2 hộp dịch vụ ngoài **Ollama (local, embedding)** và **DeepSeek API (LLM)**
  — Ollama nối vào cả 2 luồng (embed chunk / embed query), DeepSeek nối vào cuối luồng query.
- **Phía trên cùng**: 2 hộp giao diện **FastAPI** (endpoints /documents, /query, /health)
  và **Streamlit demo** (song song, Streamlit không nối vào FastAPI).
- **Viền ngoài cùng**: hộp bao "Config (.env) · Errors · Logging" như lớp cắt ngang (dashed).
- Mỗi stage vẽ hộp nhỏ với tên module (`pdf_loader.py`, `splitter.py`, `ollama_client.py`,
  `chroma_store.py`, `retriever.py`, `prompt_templates.py`, `deepseek_client.py`,
  `index_pipeline.py`, `query_pipeline.py`).
- Ghi chú trên mũi tên: chunk_id format, score = 1−distance, temperature 0.2, prefix "Instruct: …".
