"""Generate tests/fixtures/sample_tech_ebook.pdf (reproducible test fixture).

Run: uv run python scripts/make_fixture_pdf.py
"""

from pathlib import Path

import pymupdf

FIXTURE_PATH = Path("tests/fixtures/sample_tech_ebook.pdf")

PAGE_1 = """Vector Databases: A Primer

A vector database stores high-dimensional vectors and supports fast
similarity search over them. Instead of exact keyword matching, retrieval is
based on distance between embeddings, which lets systems find semantically
related items even when they share no common terms.

Embeddings are produced by machine learning models that map objects such as
text, images, or audio into a vector space where similar items are close
together. For example, the sentence "cats are pets" and "felines are
domesticated animals" would be placed near each other despite using
completely different words.

A typical RAG pipeline first indexes a corpus by embedding every chunk of
text and storing the vectors, then answers a question by embedding the query
and searching for the most similar stored vectors. The retrieved passages
are finally handed to a language model as context for generating the answer.
"""

PAGE_2 = """## Example: Python snippet

The code below demonstrates a minimal similarity search with numpy:

```python
import numpy as np

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search(query_vec, index, top_k=3):
    scores = [(cosine_similarity(query_vec, v), i) for i, v in enumerate(index)]
    scores.sort(reverse=True)
    return [i for _, i in scores[:top_k]]

vectors = np.random.rand(10, 4)
q = np.random.rand(4)
print(search(q, vectors))
```

Cosine similarity ignores the magnitude of the vectors and only measures the
angle between them, which makes it a popular metric for text embeddings that
are normalized before storage.
"""

PAGE_3 = """## Unicode and special characters

Technical documents frequently contain mathematical symbols and arrows: the
Greek letters lambda (λ) and pi (π), the arrow (→), plus common operators
such as ±, ×, and ÷. A robust text pipeline must preserve these characters
so that chunking and embedding see the original content, not replacement
glyphs.

This page exists mainly to verify that the PDF fixture round-trips these
characters through PyMuPDF without encoding corruption, which would otherwise
silently degrade retrieval quality.
"""


def main() -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    font = pymupdf.Font("cjk")  # wide-coverage fallback: Greek, arrows, CJK
    doc = pymupdf.open()
    for text in (PAGE_1, PAGE_2, PAGE_3):
        page = doc.new_page()
        page.insert_font(fontname="F0", fontbuffer=font.buffer)
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11, fontname="F0")
    doc.save(FIXTURE_PATH)
    doc.close()
    print(f"Fixture written to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
