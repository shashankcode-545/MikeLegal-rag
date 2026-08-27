"""
Chunking tests.
"""
from src.chunk import chunk_document
from src.extract import extract_pages


def test_chunk_ids_are_deterministic(sample_pdf_path):
    pages = extract_pages(sample_pdf_path)

    chunks_first_run = chunk_document(pages, doc_id="nda")
    chunks_second_run = chunk_document(pages, doc_id="nda")

    ids_first = [c.chunk_id for c in chunks_first_run]
    ids_second = [c.chunk_id for c in chunks_second_run]
    assert ids_first == ids_second
    assert len(ids_first) == len(set(ids_first)), "chunk IDs must be unique"


def test_chunk_ids_follow_expected_format(sample_pdf_path):
    pages = extract_pages(sample_pdf_path)
    chunks = chunk_document(pages, doc_id="nda")

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.chunk_id.startswith("nda::c")
        assert chunk.doc_id == "nda"


def test_chunks_carry_citation_metadata(sample_pdf_path):
    pages = extract_pages(sample_pdf_path)
    chunks = chunk_document(pages, doc_id="nda")

    for chunk in chunks:
        assert chunk.page_number >= 1
        assert isinstance(chunk.section_label, str) and chunk.section_label != ""
        assert chunk.text.strip() != ""


def test_real_nda_uses_heading_based_chunking(sample_pdf_path):
    """The sample NDA has clear '1.', '1.1', etc. numbering, so chunking
    should find real clause headings rather than falling back to fixed
    word windows."""
    pages = extract_pages(sample_pdf_path)
    chunks = chunk_document(pages, doc_id="nda")

    fallback_labeled = [c for c in chunks if c.section_label.startswith("words ")]
    assert len(fallback_labeled) == 0


def test_fixed_size_fallback_when_no_headings():
    """A document with no numbered structure at all should fall back to
    fixed-size word windows instead of producing zero/garbage chunks."""
    from src.models import Page

    plain_text = " ".join([f"word{i}" for i in range(500)])
    pages = [Page(page_number=1, text=plain_text)]

    chunks = chunk_document(pages, doc_id="plain")

    assert len(chunks) > 0
    assert all(c.section_label.startswith("words ") for c in chunks)
