"""PDF ingestion: extract per-page text with metadata using PyMuPDF."""

from pathlib import Path

import pymupdf
from pydantic import BaseModel

from src.errors import EmptyDocumentError  # re-exported for back-compat


class Document(BaseModel):
    text: str
    page_number: int  # 1-based
    source_file: str  # file name only, no directory


def load_pdf(path: str) -> list[Document]:
    """Extract text from each non-empty page of a PDF.

    Returns one `Document` per page that has text after stripping; page numbers
    are 1-based. Whitespace inside code is preserved — pages are only stripped
    at the edges.

    Raises:
        FileNotFoundError: the file does not exist (raised by PyMuPDF).
        ValueError: the file is not a valid PDF.
        EmptyDocumentError: no page contains extractable text.
    """
    pdf_path = Path(path)
    source_file = pdf_path.name
    try:
        pdf = pymupdf.open(path)
    except pymupdf.FileDataError as exc:
        raise ValueError(f"Not a valid PDF file: {path}") from exc
    except pymupdf.FileNotFoundError as exc:
        # PyMuPDF's FileNotFoundError is not the builtin one — normalize it.
        raise FileNotFoundError(path) from exc
    try:
        if not pdf.is_pdf:
            raise ValueError(f"Not a valid PDF file: {path}")
        docs: list[Document] = []
        for page_number in range(1, pdf.page_count + 1):
            text = pdf.load_page(page_number - 1).get_text("text")
            if text.strip():
                docs.append(
                    Document(
                        text=text,
                        page_number=page_number,
                        source_file=source_file,
                    )
                )
    finally:
        pdf.close()
    if not docs:
        raise EmptyDocumentError(f"No extractable text in PDF (scanned/image-only?): {path}")
    return docs
