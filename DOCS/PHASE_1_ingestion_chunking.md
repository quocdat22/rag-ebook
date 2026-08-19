# PHASE 1 — Ingestion (PDF parsing) + Chunking

**Trạng thái:** [x] Hoàn thành
**Phụ thuộc:** Phase 0
**Output:** `pdf → list[Chunk]` hoạt động, unit test đầy đủ, PDF fixture commit trong repo.

---

## 1. Mục tiêu

- Module `ingestion`: đọc PDF bằng PyMuPDF, trích text từng trang kèm metadata (SPEC 5.1).
- Module `chunking`: cắt text thành chunk ~700 token (overlap 100), **không cắt giữa code block** (SPEC 5.2, F2).
- Fixture PDF nhỏ (2–3 trang, có code block) commit vào repo để test reproducible (SPEC mục 6).

## 2. Thiết kế

### 2.1 `src/ingestion/pdf_loader.py`

Interface đúng SPEC 5.1:

```python
from pydantic import BaseModel


class Document(BaseModel):
    text: str
    page_number: int  # 1-based
    source_file: str  # tên file, không kèm đường dẫn


def load_pdf(path: str) -> list[Document]: ...
```

Implementation notes:
- `import pymupdf`; `doc = pymupdf.open(path)`; lặp `for page in doc:`; `page.get_text("text")`.
- Đóng file chắc chắn (context manager hoặc try/finally — tránh leak file handle).
- **Bỏ trang rỗng** (sau `strip()` trống → skip) để không tạo chunk rác.
- **Không normalize quá tay** — giữ whitespace trong code nguyên bản; chỉ `strip` mỗi trang. Embedding/chunking cần text gốc.
- Validate lỗi rõ ràng:
  - File không tồn tại → `FileNotFoundError` (tự nhiên của PyMuPDF, giữ nguyên).
  - File không phải PDF → raise rõ ràng.
  - PDF không có text layer / 0 trang có text → raise `EmptyDocumentError` (định nghĩa ngay trong module này; Phase 7 có thể dời về `src/errors.py`).

### 2.2 `src/chunking/splitter.py`

Interface đúng SPEC 5.2:

```python
from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    text: str
    page_number: int
    source_file: str


def split_documents(
    docs: list[Document], chunk_size: int = 700, overlap: int = 100
) -> list[Chunk]: ...
```

**Chiến lược "code-block-aware" (tự viết — mục tiêu học tập, phương án A mặc định):**

1. **Bước 1 — tách segment nguyên tử** cho từng trang:
   - Regex `r"```.*?```"` (DOTALL) tìm **fenced code block** → mỗi block là 1 segment bất khả xâm phạm.
   - Text ngoài fence: tách heading bằng regex `r"^#{1,6} .*$"` (MULTILINE) → mỗi heading là 1 segment; phần còn lại split theo `\n\n` (đoạn văn); đoạn quá dài split tiếp theo `\n`.
   - Mỗi segment mang theo (page_number, source_file) của trang nguồn.
2. **Bước 2 — gộp segment vào chunk (greedy packing):**
   - Đơn vị tính là **token ước lượng**: `estimate_tokens(text) = len(text.split()) * 1.3` (hệ số cho text kỹ thuật nhiều symbol; không cần thư viện tokenizer ở MVP — ghi chú rõ giả định này trong docstring).
   - Gộp segment liên tiếp tới khi vượt `chunk_size` → đóng chunk; lùi lại các segment cuối (đủ ~`overlap` token) làm phần đầu chunk kế tiếp.
   - **Không bao giờ cắt giữa segment** → code block, heading luôn nguyên vẹn.
   - Ngoại lệ: 1 segment đơn dài hơn `chunk_size` (code block siêu dài) → buộc phải cắt segment đó theo từng dòng (ưu tiên cắt tại `\n`), không mất nội dung.
3. **Bước 3 — tạo `Chunk`:** `chunk_id = f"{source_file}::p{page_number}::{index}"` (unique, đọc được bằng mắt).

Hàm phụ trợ trong cùng file (tách nhỏ để test độc lập):
- `split_into_segments(text: str) -> list[str]`
- `pack_segments(segments: list[str], chunk_size: int, overlap: int) -> list[list[str]]`
- `estimate_tokens(text: str) -> int`

> **Phương án B (nếu thiếu thời gian):** `uv add langchain-text-splitters`, dùng `RecursiveCharacterTextSplitter(chunk_size=..., chunk_overlap=..., separators=["\n\n", "\n", " ", ""])` — chấp nhận code block có thể bị cắt; ghi vào "giới hạn đã biết". Mặc định chọn A vì phù hợp mục tiêu học tập của đồ án (SPEC mục 3).

### 2.3 Fixture PDF — `tests/fixtures/sample_tech_ebook.pdf`

Tạo bằng script `scripts/make_fixture_pdf.py` (dùng chính pymupdf — tái tạo được):

- Trang 1: tiêu đề + 2–3 đoạn văn tiếng Anh (chủ đề kỹ thuật, ví dụ: giới thiệu vector database).
- Trang 2: heading `## Example: Python snippet` + 1 fenced code block dài (~15 dòng) + đoạn văn.
- Trang 3: heading + vài đoạn + ký tự unicode đặc biệt (`→ λ π`) để test encoding.

```bash
uv run python scripts/make_fixture_pdf.py   # sinh PDF vào tests/fixtures/
```

**PDF sinh ra được commit vào repo** (SPEC mục 6: fixture cố định, reproducible); giữ script lại để tái tạo khi cần.

## 3. Test chi tiết

### `tests/unit/test_pdf_loader.py`

| Test | Assert |
|---|---|
| `test_load_fixture_returns_3_pages` | `len(docs) == 3`; `page_number` = 1, 2, 3 đúng thứ tự |
| `test_text_non_empty` | mọi doc có `len(doc.text.strip()) > 0` |
| `test_source_file_is_basename` | `source_file == "sample_tech_ebook.pdf"` (không kèm đường dẫn) |
| `test_unicode_preserved` | text trang 3 chứa `"→"`, `"λ"` (encoding không vỡ) |
| `test_missing_file_raises` | `pytest.raises(FileNotFoundError)` với path không tồn tại |
| `test_non_pdf_raises` | file text thường → exception rõ ràng (không phải traceback khó hiểu) |

### `tests/unit/test_splitter.py`

Dùng text giả lập tự build trong test (không phụ thuộc PDF fixture):

| Test | Assert |
|---|---|
| `test_chunks_within_size` | mọi chunk: `estimate_tokens(chunk.text) <= chunk_size` |
| `test_code_block_not_split` | chunk chứa code: fence ```` ``` ```` xuất hiện theo **cặp** (mở + đóng) trong cùng chunk |
| `test_overlap` | input đủ dài: phần đuôi chunk[i] trùng phần đầu chunk[i+1] xấp xỉ `overlap` token |
| `test_metadata_preserved` | `page_number`/`source_file` của mọi chunk đúng theo doc nguồn |
| `test_chunk_ids_unique` | không có 2 chunk trùng `chunk_id` |
| `test_empty_input` | `split_documents([], ...) == []` |
| `test_single_long_segment_split_at_newline` | segment dài hơn chunk_size bị cắt tại `\n`, không mất nội dung |

## 4. Tiêu chí hoàn thành (DoD)

- [x] Chạy tay: `uv run python -c "from src.ingestion.pdf_loader import load_pdf; from src.chunking.splitter import split_documents; docs = load_pdf('tests/fixtures/sample_tech_ebook.pdf'); chunks = split_documents(docs); print(len(docs), len(chunks))"` → in số trang + số chunk hợp lý (3 trang, 3 chunk)
- [x] `uv run pytest tests/unit/test_pdf_loader.py tests/unit/test_splitter.py` → toàn bộ pass
- [x] `uv run ruff check --fix . && uv run ruff format . && uv run pytest` → 3 lệnh pass
- [x] Fixture PDF + script tạo fixture đã commit

## 5. Rủi ro & lưu ý

- **PDF 2 cột / bảng:** `get_text("text")` có thể lẫn cột, trộn dòng — ghi nhận giới hạn, không xử lý ở MVP (SPEC mục 9).
- **PDF scan (không text layer):** không OCR ở MVP — `load_pdf` trả rỗng → `EmptyDocumentError` với message hướng dẫn rõ ràng.
- **Ước lượng token thô** (word × 1.3): lệch ±15% so với token thật — chấp nhận cho MVP; cô lập trong `estimate_tokens()` để dễ thay bằng tokenizer thật sau.
- **Code block indent-based** (không có fence ```) chưa được bảo vệ — ghi vào "giới hạn đã biết" (SPEC mục 9); có thể thêm pattern indent-block ở Phase 7 nếu còn thời gian.
