# PHASE 7 — Polish & chuẩn bị trình bày

**Trạng thái:** [x] Hoàn thành
**Phụ thuộc:** Phase 5 (và Phase 6 nếu làm UI)
**Output:** Repo sẵn sàng trình bày: README đầy đủ, error handling + logging nhất quán, toàn bộ check pass.

---

## 1. Mục tiêu

- README hoàn chỉnh (cài đặt, chạy, kiến trúc, giới hạn đã biết).
- Logging + hệ thống exception nhất quán.
- Type checking (mypy), lint, format, test — toàn bộ pass.
- (Optional) Docker compose cho Ollama + app.

## 2. Công việc chi tiết

### 2.1 Logging

- `src/logging_config.py` (hoặc cấu hình trong `config.py`): `logging.basicConfig` format `%(asctime)s %(levelname)s %(name)s: %(message)s`, level theo env `LOG_LEVEL` (mặc định INFO — thêm vào `.env.example`).
- Pipeline log các mốc chính, đủ để debug khi demo gặp sự cố:
  - index: `loaded N pages` → `split into N chunks` → `embedded N texts in Xs` → `added N chunks to collection`
  - query: `retrieved K chunks in Xs` → `generation done in Xs`

### 2.2 Error handling nhất quán

Gom exception vào `src/errors.py` (dời các class đã định nghĩa rải rác về đây):

| Exception | Ý nghĩa | Nơi raise |
|---|---|---|
| `RagEbookError` (base) | lỗi chung của hệ thống | — |
| `EmptyDocumentError` | PDF không có text layer / 0 trang text | ingestion |
| `OllamaUnavailableError` | Ollama không chạy / timeout | embedding |
| `GenerationError` | DeepSeek lỗi / timeout / empty | generation |
| `ConfigurationError` | thiếu API key, config sai | config/pipeline |

- FastAPI: `@app.exception_handler` map các lỗi trên → HTTP 503/400 với message rõ ràng (không trả traceback trần).
- Mọi message lỗi phải **hành động được**: *"chạy `ollama serve`"*, *"kiểm tra DEEPSEEK_API_KEY trong .env"*, không chỉ mô tả sự cố.

### 2.3 README.md

Các mục bắt buộc:

1. Giới thiệu + tính năng MVP (bám SPEC mục 1).
2. **Kiến trúc**: sơ đồ 2 luồng indexing/query (vẽ lại từ SPEC mục 4).
3. Cài đặt: prerequisites (Python 3.12, Ollama, `ollama pull qwen3-embedding:0.6b`), `uv sync`, tạo `.env`.
4. Chạy: CLI / API (ví dụ curl) / Streamlit (nếu có Phase 6).
5. Test: `uv run pytest`, giải thích marker `integration`.
6. Cấu hình: bảng env vars (từ `.env.example`).
7. **Giới hạn đã biết** (SPEC mục 9): không OCR, không rerank, chunking chưa bảo vệ indent-block, chi phí DeepSeek theo giờ cao/thấp điểm.
8. Hướng phát triển v2+ (SPEC mục 10) — ngắn gọn.

### 2.4 Chất lượng code

```bash
uv run mypy src            # fix lỗi type (hoặc ignore có lý do ghi rõ)
uv run ruff check --fix .
uv run ruff format .
uv run pytest              # toàn bộ, kể cả integration nếu có service thật
```

### 2.5 (Optional) Docker

- `Dockerfile` cho app (python:3.12-slim + uv sync).
- `docker-compose.yml`: service `ollama` (image `ollama/ollama`, volume model) + `app` (build ., env từ .env, port 8000), `depends_on: ollama`.
- **Trong container, `OLLAMA_HOST` phải là `http://ollama:11434`** (tên service, không phải localhost).
- Thiếu thời gian → bỏ qua, ghi rõ trong README là chạy local không container.

### 2.6 Demo script (hữu ích cho báo cáo)

`scripts/demo.py`: chạy 3–4 câu hỏi mẫu lên fixture PDF thật, in answer + citations để chụp màn hình báo cáo. Lưu ý mỗi lần chạy tốn 1 ít phí DeepSeek.

## 3. Tiêu chí hoàn thành (DoD)

- [x] README đủ 8 mục trên; người khác clone về làm theo chạy được
- [x] Thử tắt Ollama / xoá API key → lỗi hiển thị đúng message hướng dẫn
- [x] `uv run ruff check --fix .` → `uv run ruff format .` → `uv run pytest` → `uv run mypy src`: **4 lệnh pass**
- [ ] (Nếu làm Docker) `docker compose up` chạy được end-to-end — **bỏ qua Docker** (ghi rõ trong README: chạy local, không container)
- [x] Đã đọc trang pricing DeepSeek chính thức, ước tính được chi phí cho buổi demo (SPEC mục 9)
- [x] Commit cuối + tag (nếu dùng git): `v0.1.0-mvp`

## 4. Rủi ro & lưu ý

- **Đừng "polish" quá sâu:** MVP để trình bày + học; các mục v2 (rerank, RAGAS, OCR...) ghi vào README "Hướng phát triển" là đủ.
- **Check chi phí trước demo:** chạy thử ~20–30 câu hỏi để biết chi phí thực tế trước khi trình bày.
- **`uv.lock` phải commit** và `uv sync` chạy sạch (SPEC 3.1) — nếu có máy khác/CI thì kiểm tra luôn.
