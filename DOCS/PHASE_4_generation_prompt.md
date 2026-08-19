# PHASE 4 — Generation (DeepSeek) + Prompt template

**Trạng thái:** [ ] Chưa bắt đầu
**Phụ thuộc:** Phase 2 (dùng `RetrievedChunk` để build context)
**Output:** `context + question → answer` có citation marker `[n]`, unit test với mock API (không tốn tiền).

---

## 1. Mục tiêu

- `DeepSeekClient` gọi DeepSeek API qua OpenAI SDK (`base_url=https://api.deepseek.com`) (SPEC 5.6, F5).
- Prompt template: **system prompt cố định** (tận dụng cache-hit pricing của DeepSeek — SPEC 5.6) + user prompt = context đánh số + câu hỏi.
- Hàm trích citation `[n]` từ answer (SPEC 5.6, F6).

## 2. Thiết kế

### 2.1 `src/generation/prompt_templates.py`

```python
import re

from src.vectorstore.chroma_store import RetrievedChunk

SYSTEM_PROMPT: str = (
    "You are a technical assistant that answers questions strictly based on the "
    "provided context. If the context does not contain the answer, say so explicitly. "
    "Always cite sources inline using [n] where n is the context number, e.g. [1] or [2][3]. "
    "Never invent sources."
)
# KHÔNG thay đổi chuỗi này giữa các request → DeepSeek prefix cache hit (giảm chi phí đáng kể).
# Mọi thứ thay đổi theo câu hỏi phải nằm ở USER prompt (SPEC 5.6).


def build_user_prompt(context_chunks: list[RetrievedChunk], question: str) -> str: ...


CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_cited_indices(answer: str) -> list[int]:
    """Các index context được trích dẫn trong answer, theo thứ tự xuất hiện, không trùng."""
```

Format user prompt (context đánh số, question ở cuối):

```text
Context:
[1] (sample_tech_ebook.pdf, page 2)
<text>

[2] (sample_tech_ebook.pdf, page 3)
<text>

Question: What is a vector database?

Answer with inline citations like [1].
```

### 2.2 `src/generation/deepseek_client.py`

```python
from typing import Protocol

from openai import OpenAI


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class GenerationError(RuntimeError):
    """Lỗi khi gọi/tạo response từ DeepSeek (timeout, rate limit, empty...)."""


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout: float = 60.0,
        client: OpenAI | None = None,  # để test inject fake client
    ): ...
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
```

Implementation notes:
- Dùng SDK `openai` (OpenAI-compatible): `OpenAI(api_key=..., base_url=...)`.
- Gọi: `client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.2)` — nhiệt độ thấp vì RAG cần trả lời bám context, không sáng tạo.
- Trả về `choice.message.content`; None/empty → `GenerationError`.
- Bọc lỗi: `openai.APITimeoutError`, `APIConnectionError`, `RateLimitError` (429) → raise `GenerationError` kèm message gốc (SPEC 5.6).
- **Model:** dùng thẳng `deepseek-v4-flash` (mặc định) / `deepseek-v4-pro`; các alias cũ `deepseek-chat` / `deepseek-reasoner` đã bị khai tử — **không dùng**.

## 3. Test chi tiết

### `tests/unit/test_prompt_templates.py`

| Test | Assert |
|---|---|
| `test_user_prompt_contains_context_and_question` | question nằm cuối; mọi chunk text + `(file, page N)` xuất hiện |
| `test_numbering_starts_at_1` | context bắt đầu bằng `[1]`, tăng dần đúng thứ tự chunk |
| `test_system_prompt_is_constant` | `SYSTEM_PROMPT` là hằng string, không phải hàm |
| `test_extract_citations` | `"See [1] and [2][3]."` → `[1, 2, 3]`; không có → `[]`; `[99]` vẫn trả 99 (validate range ở Phase 5) |

### `tests/unit/test_deepseek_client.py` — mock OpenAI, **không gọi API thật**

Inject fake client qua constructor:

| Test | Assert |
|---|---|
| `test_generate_returns_content` | fake trả content `"The answer is 42."` → generate trả đúng |
| `test_messages_format` | fake ghi lại args → 2 messages, system == `SYSTEM_PROMPT`, user chứa context + question |
| `test_model_passed` | gọi `create` với `model=` đúng giá trị constructor |
| `test_empty_response_raises` | fake trả `content=None` → `GenerationError` |
| `test_timeout_raises` | fake raise `openai.APITimeoutError` → `GenerationError` |
| `test_rate_limit_raises` | fake raise `openai.RateLimitError` → `GenerationError` |

## 4. Kiểm tra thủ công (1 lần duy nhất, dùng API thật)

```bash
uv run python scripts/smoke_generate.py
# Build 1 context giả (vài dòng về fixture PDF) + 1 câu hỏi → in answer.
# Kỳ vọng: câu trả lời tiếng Anh, có marker [n], tốn vài xu.
```

## 5. Tiêu chí hoàn thành (DoD)

- [ ] Unit test pass — không tốn tiền API
- [ ] Smoke generate 1 lần bằng API thật → answer có `[n]`
- [ ] `uv run ruff check --fix . && uv run ruff format . && uv run pytest` → pass

## 6. Rủi ro & lưu ý

- **Chi phí:** DeepSeek tính phí theo giờ cao điểm/thấp điểm (SPEC mục 9) — đọc trang pricing chính thức trước khi ước tính chi phí demo. System prompt cố định giúp cache-hit giảm đáng kể chi phí lặp lại.
- **Temperature 0.2** có thể chỉnh; đừng để model "sáng tạo" ngoài context — system prompt đã ép buộc.
- **Rate limit khi gọi dồn dập** (demo liên tục) → 429; thêm retry đơn giản (1 lần, backoff 2–5s) nếu gặp — hoặc để Phase 7.
- **Phase này chạy song song với Phase 3 được** (chỉ phụ thuộc Phase 2 qua kiểu `RetrievedChunk`).
