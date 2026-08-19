# PHASE 6 — UI (Streamlit) — OPTIONAL

**Trạng thái:** [x] Hoàn thành
**Phụ thuộc:** Phase 5
**Output:** Demo trực quan: upload PDF → hỏi → xem answer + citation.

> **Có thể bỏ qua phase này** nếu ưu tiên thời gian — Swagger UI (`/docs`) của FastAPI đã đủ để demo (SPEC 5.9). Nếu bỏ: nhảy thẳng Phase 7 và ghi rõ trong README rằng UI chưa có.

---

## 1. Mục tiêu

- Streamlit app đơn giản, 1 file: upload PDF, nhập câu hỏi, hiện answer + danh sách citation (file, trang, đoạn trích) (SPEC 5.9).

## 2. Thiết kế — `src/ui/streamlit_app.py`

```bash
uv add streamlit            # chỉ cài trong phase này
uv run streamlit run src/ui/streamlit_app.py
```

Cấu trúc app:

1. **Init resources 1 lần** (`@st.cache_resource`): tạo `OllamaEmbeddingClient`, `ChromaVectorStore("rag_ebook", persist_dir)`, `DeepSeekClient` từ `settings` — không tạo lại mỗi lần rerun.
2. **Tab "Ingest"**: `st.file_uploader(type=["pdf"])` → lưu tạm `data/uploads/` → gọi `IndexPipeline.run` → `st.success(f"Indexed {n} chunks")`; lỗi → `st.error` với message rõ ràng.
3. **Tab "Ask"**: `st.chat_input` / `st.text_input` câu hỏi → `QueryPipeline.run` → `st.markdown(answer)`; citations trong `st.expander("Sources")`: mỗi dòng `📄 {source_file} — p.{page_number}` + đoạn trích (giới hạn ~300 ký tự).
4. **Sidebar**: số chunk đã index (đếm từ collection), slider `top_k`, slider `min_score`
   (ngưỡng similarity, mặc định theo `MIN_SCORE` trong settings), model đang dùng.

**Không kết nối Streamlit với FastAPI** — app gọi thẳng pipeline layer (cùng process). Streamlit là demo; FastAPI là interface "thật".

## 3. Kiểm tra (bằng tay — không viết unit test cho UI, Streamlit khó test tự động)

- [x] Upload fixture PDF → hiện thông báo số chunk đã index
- [x] Hỏi câu liên quan nội dung → trả lời kèm `[n]`, Sources hiện đúng trang
- [x] Hỏi câu lạc đề → model trả lời "không có trong context"
- [x] Tắt Ollama rồi ingest → hiện lỗi rõ ràng, app không crash

> Kiểm chứng tự động (headless) bằng `streamlit.testing.v1.AppTest` + service thật:
> `uv run python scripts/smoke_ui.py` — boot OK, ingest 3 chunk, answer có
> `Sources (1)`, câu lạc đề → 0 sources.

## 4. Tiêu chí hoàn thành (DoD)

- [x] Demo tay toàn bộ luồng OK (chụp ảnh/video cho báo cáo)
- [x] `uv run ruff check . && uv run pytest` vẫn pass (UI không phá test cũ)

## 5. Rủi ro & lưu ý

- `st.cache_resource` giữ client sống lâu → sửa `.env` phải restart app (ghi chú nhỏ trong UI).
- **Không deploy công khai** (không có auth — SPEC mục 9): chỉ chạy local `localhost:8501`.
- Không dùng thời gian quý vào chỉnh đẹp UI — MVP là demo, ưu tiên đúng chức năng.
