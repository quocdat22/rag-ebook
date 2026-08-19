from src.config import Settings


def test_settings_defaults():
    s = Settings(_env_file=None)  # không đọc .env thật để test không phụ thuộc máy
    assert s.ollama_embed_model == "qwen3-embedding:0.6b"
    assert s.chunk_size == 700
    assert s.chunk_overlap == 100
    assert s.deepseek_model == "deepseek-v4-flash"
    assert s.min_score == 0.3
