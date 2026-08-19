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
    min_score: float = 0.3  # cosine similarity; MIN_SCORE=0.0 để tắt lọc

    # Logging
    log_level: str = "INFO"


settings = Settings()  # singleton dùng chung toàn app
