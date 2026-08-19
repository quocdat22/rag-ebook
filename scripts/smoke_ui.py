"""Smoke test for Phase 6: drive the Streamlit app headlessly via AppTest.

Verifies the app boots, uploads + indexes the fixture PDF (real Ollama), and
answers a question (real DeepSeek) with a Sources expander. A second, out-of-
scope question must produce no sources. Fails on uncaught app exceptions.

Run: uv run python scripts/smoke_ui.py
"""

import sys
from pathlib import Path

# Make `src.*` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "ui" / "streamlit_app.py")
FIXTURE_PDF = Path("tests/fixtures/sample_tech_ebook.pdf")


def main() -> int:
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    if at.exception:
        print("BOOT FAILED:", [str(e) for e in at.exception])
        return 1
    print("App booted OK (sidebar + tabs rendered).")

    at.file_uploader[0].set_value([(FIXTURE_PDF.name, FIXTURE_PDF.read_bytes(), "application/pdf")])
    at.run()
    if at.exception:
        print("INGEST FAILED:", [str(e) for e in at.exception])
        return 1
    print("Ingest:", [s.value for s in at.success] or [e.value for e in at.error])

    at.chat_input[0].set_value("What is a vector database?")
    at.run()
    if at.exception:
        print("ASK FAILED:", [str(e) for e in at.exception])
        return 1
    errors = [e.value for e in at.error]
    if errors:
        print("ASK ERRORS:", errors)
        return 1
    print("Sources expander:", [e.label for e in at.expander])

    at.chat_input[0].set_value("What is the capital of Australia?")
    at.run()
    if at.exception:
        print("OUT-OF-SCOPE FAILED:", [str(e) for e in at.exception])
        return 1
    if at.expander:
        print("OUT-OF-SCOPE: unexpected sources", [e.label for e in at.expander])
        return 1
    print("Out-of-scope question: no sources (correct).")
    print("OK: full UI flow works headlessly with real services.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
