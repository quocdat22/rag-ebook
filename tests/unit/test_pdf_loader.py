"""Unit tests for src.ingestion.pdf_loader."""

from pathlib import Path

import pymupdf
import pytest

from src.ingestion.pdf_loader import EmptyDocumentError, load_pdf

FIXTURE = Path("tests/fixtures/sample_tech_ebook.pdf")


def test_load_fixture_returns_3_pages():
    docs = load_pdf(str(FIXTURE))
    assert [d.page_number for d in docs] == [1, 2, 3]


def test_text_non_empty():
    docs = load_pdf(str(FIXTURE))
    assert all(doc.text.strip() for doc in docs)


def test_source_file_is_basename():
    docs = load_pdf(str(FIXTURE))
    assert all(doc.source_file == "sample_tech_ebook.pdf" for doc in docs)


def test_unicode_preserved():
    docs = load_pdf(str(FIXTURE))
    page_3 = next(doc for doc in docs if doc.page_number == 3)
    assert "→" in page_3.text
    assert "λ" in page_3.text


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_pdf(str(FIXTURE.parent / "does_not_exist.pdf"))


def test_non_pdf_raises(tmp_path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("just some text, not a pdf")
    with pytest.raises(ValueError, match="Not a valid PDF"):
        load_pdf(str(text_file))


def test_empty_pdf_raises(tmp_path):
    empty_pdf = tmp_path / "empty.pdf"
    doc = pymupdf.open()
    doc.new_page()  # blank page, no text
    doc.save(str(empty_pdf))
    doc.close()
    with pytest.raises(EmptyDocumentError):
        load_pdf(str(empty_pdf))
