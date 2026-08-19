# PHASE 0 — Setup môi trường & repo skeleton

**Trạng thái:** [x] Hoàn thành
**Phụ thuộc:** Không
**Output:** Repo skeleton chạy được `uv run pytest` (pass), Ollama local với `qwen3-embedding:0.6b` hoạt động, module config `.env` dùng được.

---

## 1. Mục tiêu

- Khởi tạo project Python 3.12 với `uv` (`pyproject.toml` + `uv.lock`).
- Cài Ollama + pull model embedding `qwen3-embedding:0.6b` (1024-dim).
- Cài đầy đủ dependency runtime + dev; cấu hình pytest / ruff / mypy.
- Tạo cấu trúc thư mục `src/`, `tests/`, `.env.example`, `.gitignore`.
- Viết `src/config.py` (pydantic-settings) — nền tảng cho mọi phase sau.

## 2. Công việc chi tiết

### 2.1 Cài Ollama + model embedding (local)

```bash
# Cài Ollama (Linux/WSL2)
curl -fsSL https://ollama.com/install.sh | sh

# Pull model embedding 1024-dim (0.6B, context 32k)
ollama pull qwen3-embedding:0.6b

# Kiểm tra model đã có
ollama list
```

Kiểm tra API trả đúng vector 1024 chiều:

```bash
curl http://localhost:11434/api/embeddings \
  -d '{"model":"qwen3-embedding:0.6b","prompt":"hello world"}' | head -c 300
# Kỳ vọng: JSON có "embedding": [ ...1024 số... ]
```

> `qwen3-embedding` là model **instruction-aware** — phía query nên prepend task instruction (`Instruct: ...\nQuery: ...`) để tăng chất lượng retrieval; phía document không cần. Chi tiết triển khai ở Phase 2.

> Nếu Ollama chưa chạy: `ollama serve` (trên Linux có systemd service tự start).

### 2.2 Khởi tạo project với `uv`

```bash
# Trong thư mục rag_ebook:
uv init                          # tạo pyproject.toml + README.md tại thư mục hiện tại
uv venv --python 3.12            # tạo .venv dùng Python 3.12 (uv tự tải nếu máy chưa có)
uv python pin 3.12               # ghi requires-python vào pyproject.toml
```

### 2.3 Thêm dependency (chỉ dùng `uv add`, không sửa tay pyproject.toml)

```bash
# Runtime
uv add fastapi uvicorn pydantic-settings chromadb pymupdf openai httpx python-multipart
#  - httpx: HTTP client cho Ollama client (Phase 2), cũng dùng trong TestClient
#  - python-multipart: cần cho FastAPI UploadFile (Phase 5) — cài sẵn từ giờ

# Dev
uv add --dev pytest pytest-mock mypy ruff
```

Ghi chú dependency:
- `pymupdf` import trong code là `import pymupdf` (alias `fitz` vẫn chạy).
- `langchain-text-splitters` **chưa cài** — Phase 1 ưu tiên tự viết splitter (mục tiêu học tập). Chỉ `uv add langchain-text-splitters` nếu chọn phương án B.
- `streamlit` cài ở Phase 6 (không cần sớm).

### 2.4 Cấu hình pytest / ruff / mypy trong `pyproject.toml`

Thêm block sau (đây là cấu hình, sửa pyproject.toml hợp lệ vì không phải thêm dependency):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]                # để test import được `src.*`
markers = [
    "integration: cần service thật (Ollama/DeepSeek), bỏ qua trong CI",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
```

### 2.5 Cấu trúc thư mục skeleton

Tạo cây thư mục theo SPEC mục 5 (các `__init__.py` để trống):

```text
rag_ebook/
├── src/
│   ├── ingestion/__init__.py
│   ├── chunking/__init__.py
│   ├── embedding/__init__.py
│   ├── vectorstore/__init__.py
│   ├── retrieval/__init__.py
│   ├── generation/__init__.py
│   ├── pipeline/__init__.py
│   ├── api/__init__.py
│   ├── ui/__init__.py
│   └── config.py              ← viết ngay trong phase này
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/              ← Phase 1 mới có file
├── scripts/                   ← script tiện ích (tạo fixture PDF, smoke test, demo)
├── data/                      ← chroma persist + uploads (gitignore)
├── conftest.py                ← rỗng, giúp pytest nhận project root làm sys.path
├── .env.example
├── .gitignore
├── pyproject.toml
└── uv.lock                    ← commit vào repo
```

### 2.6 `.env.example` + `.gitignore`

`.env.example` (nội dung từ SPEC mục 7):

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b

DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com

CHROMA_PERSIST_DIR=./data/chroma
CHUNK_SIZE=700
CHUNK_OVERLAP=100
TOP_K=5
MIN_SCORE=0.3   # ngưỡng similarity tối thiểu (0.0 = tắt lọc)
```

```bash
cp .env.example .env    # rồi điền DEEPSEEK_API_KEY thật (tạo tại platform.deepseek.com)
```

`.gitignore`:

```gitignore
.venv/
.env
data/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
dist/
```

### 2.7 `src/config.py` — module config trung tâm

```python
"""Central configuration via pydantic-settings. Values come from .env / environment."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_embed_model: str = "qwen3-embedding:0.6b"

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com"

    # Vector store
    chroma_persist_dir: str = "./data/chroma"

    # Chunking & retrieval
    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k: int = 5


settings = Settings()  # singleton dùng chung toàn app
```

> `Settings()` đọc file `.env` ở thư mục làm việc hiện tại → luôn chạy `uv run ...` từ project root.

### 2.8 Smoke test đầu tiên

`tests/unit/test_config.py`:

```python
from src.config import Settings


def test_settings_defaults():
    s = Settings(_env_file=None)  # không đọc .env thật để test không phụ thuộc máy
    assert s.ollama_embed_model == "qwen3-embedding:0.6b"
    assert s.chunk_size == 700
    assert s.chunk_overlap == 100
    assert s.deepseek_model == "deepseek-v4-flash"
```

### 2.9 Khởi tạo git (nếu chưa có)

```bash
git init
git add .
git commit -m "Phase 0: project skeleton + config"   # sau khi các check ở mục 4 pass
```

## 3. Files tạo mới

| File | Nội dung |
|---|---|
| `pyproject.toml` | metadata + dependency + config pytest/ruff/mypy |
| `uv.lock` | lock dependency (commit) |
| `src/config.py` | `Settings` (pydantic-settings) + singleton |
| `src/*/__init__.py` | 9 package rỗng |
| `tests/unit/test_config.py` | smoke test config |
| `.env.example`, `.env`, `.gitignore`, `conftest.py` | config & housekeeping |

## 4. Tiêu chí hoàn thành (DoD)

- [x] `uv run python -c "from src.config import settings; print(settings.ollama_embed_model)"` in `qwen3-embedding:0.6b`
- [x] `uv run pytest` → **pass** (≥1 test xanh)
- [x] `uv run ruff check --fix . && uv run ruff format .` → không lỗi
- [x] `ollama list` có `qwen3-embedding:0.6b`; curl `/api/embeddings` trả 1024 số
- [x] `.env` đã có `DEEPSEEK_API_KEY` thật
- [x] Commit đầu tiên hoàn tất

## 5. Rủi ro & lưu ý

- **WSL2 + Ollama:** Ollama chạy trong WSL2 OK; GPU pass-through cần WSL2 mới + driver NVIDIA. CPU-only vẫn đủ cho `qwen3-embedding:0.6b` (639MB, chỉ chậm hơn chút).
- **Python 3.12:** nếu máy chưa có, `uv venv --python 3.12` tự tải về — cần mạng.
- **ChromaDB version:** pin theo `uv.lock`; không nâng cấp giữa chừng nếu không cần (API có thể đổi giữa các bản chromadb).
- **`uv python pin`** tạo file `.python-version` — commit luôn file này để máy khác/CI dùng đúng Python.
