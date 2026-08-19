"""Streamlit demo app (SPEC 5.9): upload PDF -> ask a question -> answer + citations.

Run:
    uv run streamlit run src/ui/streamlit_app.py   # open http://localhost:8501

Local-only demo (no auth — SPEC mục 9): do not deploy publicly. The app talks
to the pipeline layer directly in the same process; it does NOT connect to the
FastAPI server — FastAPI stays the "real" interface.
"""

import sys
from pathlib import Path

# Make `src.*` importable when streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.config import Settings
from src.embedding.ollama_client import OllamaEmbeddingClient, OllamaUnavailableError
from src.generation.deepseek_client import DeepSeekClient, GenerationError
from src.pipeline.index_pipeline import IndexPipeline
from src.pipeline.query_pipeline import QueryPipeline
from src.vectorstore.chroma_store import ChromaVectorStore

UPLOAD_DIR = Path("data/uploads")


@st.cache_resource
def get_resources():
    """Build the shared services once per process (edit .env -> restart app)."""
    settings = Settings()
    store = ChromaVectorStore("rag_ebook", persist_dir=settings.chroma_persist_dir)
    embedder = OllamaEmbeddingClient(model=settings.ollama_embed_model, host=settings.ollama_host)
    llm = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        base_url=settings.deepseek_base_url,
    )
    index_pipeline = IndexPipeline(embedder, store, settings.chunk_size, settings.chunk_overlap)
    return settings, store, embedder, llm, index_pipeline


def main() -> None:
    st.set_page_config(page_title="rag-ebook demo", page_icon="📚", layout="wide")
    settings, store, embedder, llm, index_pipeline = get_resources()

    st.sidebar.header("📚 rag-ebook")
    st.sidebar.caption("Local RAG demo — Ollama embedding + DeepSeek generation")
    st.sidebar.write(f"Collection: `{store.collection_name}`")
    st.sidebar.write(f"Chunks indexed: **{store.count()}**")
    st.sidebar.write(f"Embedding model: `{settings.ollama_embed_model}`")
    st.sidebar.write(f"LLM model: `{settings.deepseek_model}`")
    top_k = st.sidebar.slider("top_k", min_value=1, max_value=10, value=settings.top_k)
    st.sidebar.caption("Đổi `.env` → restart app (client được cache)")

    tab_ingest, tab_ask = st.tabs(["📤 Ingest PDF", "💬 Ask a question"])

    with tab_ingest:
        st.subheader("Upload a technical PDF (English)")
        uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded is not None:
            with st.spinner("Indexing… (embedding may take a while)"):
                try:
                    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                    dest = UPLOAD_DIR / uploaded.name
                    dest.write_bytes(uploaded.getvalue())
                    chunks_indexed = index_pipeline.run(str(dest))
                except OllamaUnavailableError as exc:
                    st.error(f"Ollama is not running: {exc}")
                except (ValueError, FileNotFoundError) as exc:
                    st.error(str(exc))
                else:
                    st.success(f"✅ Indexed {chunks_indexed} chunks from `{uploaded.name}`")

    with tab_ask:
        st.subheader("Ask about the indexed books")
        question = st.chat_input("Ask a question about the uploaded book…")
        if question:
            query_pipeline = QueryPipeline(embedder, store, llm, top_k)
            with st.spinner("Thinking…"):
                try:
                    result = query_pipeline.run(question)
                except GenerationError as exc:
                    st.error(f"Generation failed: {exc}")
                except OllamaUnavailableError as exc:
                    st.error(f"Ollama is not running: {exc}")
                else:
                    st.markdown(result.answer)
                    if result.citations:
                        with st.expander(f"Sources ({len(result.citations)})"):
                            for citation in result.citations:
                                st.markdown(
                                    f"📄 **{citation.source_file}** — p.{citation.page_number} "
                                    f"· `{citation.chunk_id}`"
                                )
                                preview = citation.text[:300]
                                if len(citation.text) > 300:
                                    preview += "…"
                                st.text(preview)


if __name__ == "__main__":
    main()
