# PHASE 3 — Retrieval

**Trạng thái:** [ ] Chưa bắt đầu
**Phụ thuộc:** Phase 2 (dùng `EmbeddingClient`, `VectorStore`, `RetrievedChunk`)
**Output:** `question → list[RetrievedChunk]` đúng top-k, có unit test với fake đôi (test double).

---

## 1. Mục tiêu

- Module `retriever` ghép embedding + vectorstore thành bước retrieve (SPEC 5.5, F4).
- Hỗ trợ lọc theo ngưỡng similarity tối thiểu (chống trả context không liên quan vào prompt).

## 2. Thiết kế — `src/retrieval/retriever.py`

Interface đúng SPEC 5.5:

```python
from src.embedding.ollama_client import EmbeddingClient
from src.vectorstore.chroma_store import RetrievedChunk, VectorStore


def retrieve(
    question: str,
    embedder: EmbeddingClient,
    store: VectorStore,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[RetrievedChunk]: ...
```

Luồng (bám sơ đồ SPEC 4.2):

1. `query_embedding = embedder.embed_query(question)` — `embed_query` lo prefix instruction (model `qwen3-embedding:0.6b` instruction-aware, SPEC 5.3).
2. `results = store.query(query_embedding, top_k=top_k)`
3. Nếu `min_score > 0`: chỉ giữ kết quả có `score >= min_score`.
4. Trả về list **sắp theo score giảm dần** (sort lại trong hàm, không tin store đã sắp).

Thiết kế đơn giản: **hàm thuần nhận dependency qua tham số** (dễ test fake đôi nhất). Nếu sau này cần thêm cache/rerank thì bọc thành class `Retriever` với cùng kiểu dependency injection.

## 3. Test chi tiết — `tests/unit/test_retriever.py`

Fake đôi định nghĩa ngay trong file test (đơn giản hơn `unittest.mock` khi fake class thuần đã đủ):

```python
class FakeEmbedder:
    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return [self.vector] * len(texts)

    def embed_query(self, text):
        self.calls.append(text)
        return self.vector


class FakeStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def query(self, query_embedding, top_k=5):
        self.calls.append((query_embedding, top_k))
        return self.results[:top_k]

    def add(self, *args, **kwargs):
        pass
```

| Test | Assert |
|---|---|
| `test_calls_embedder_then_store` | `FakeEmbedder.calls == ["the question"]` (qua `embed_query`); `FakeStore.calls[0][1] == 5` (đúng thứ tự gọi, đúng top_k) |
| `test_question_embedded_exactly_once` | `embed_query` chỉ được gọi đúng 1 lần với câu hỏi gốc |
| `test_returns_expected_count` | len kết quả == min(top_k, số kết quả store có) |
| `test_top_k_passed_through` | `top_k=3` → store nhận 3 |
| `test_min_score_filters` | results score `[0.9, 0.2, 0.1]`, `min_score=0.5` → chỉ còn 1 |
| `test_empty_store_ok` | store trả `[]` → trả `[]`, không raise |
| `test_sort_by_score` | store trả lộn xộn → kết quả giảm dần theo score |

## 4. Kiểm tra thủ công (smoke, dùng collection đã index ở Phase 2)

```bash
uv run python scripts/smoke_retrieve.py "What is a vector database?"
# Kỳ vọng: in top-5 chunk (page, source, score, ~100 ký tự đầu)
```

Đánh giá bằng mắt: chunk trả về có liên quan câu hỏi không?
- Không liên quan → thử tăng `chunk_size` / đổi cách tách (quay lại Phase 1), hoặc ghi nhận để cân nhắc re-ranking ở v2 (SPEC mục 9).

## 5. Tiêu chí hoàn thành (DoD)

- [ ] Unit test pass — **không cần Ollama/Chroma thật**
- [ ] Smoke retrieve bằng Ollama thật trả kết quả hợp lý với câu hỏi về nội dung fixture PDF
- [ ] `uv run ruff check --fix . && uv run ruff format . && uv run pytest` → pass

## 6. Rủi ro & lưu ý

- **Chất lượng retrieval phụ thuộc hoàn toàn embedding similarity** (không rerank ở MVP) — nếu câu trả lời cuối tệ, đây là điểm nghi ngờ đầu tiên (SPEC mục 9).
- **Model instruction-aware:** câu hỏi phải được embed bằng `embed_query()` (có prefix `Instruct: ...`) — nếu vô tình dùng `embed()` không prefix cho query, retrieval sẽ giảm chất lượng 1–5%.
- `min_score` để mặc định 0.0; chỉ bật khi thấy câu trả lời lẫn thông tin ngoài tài liệu.
- Phase này code lượng nhỏ — thời gian còn dư nên dành để viết thêm test edge case và đọc lại Phase 1/2 (chunking là nơi ảnh hưởng chất lượng lớn nhất).
