"""
PDF text extraction.

Extracts page-by-page so downstream chunking can tag every passage with
its source page number.
"""
from pathlib import Path
from typing import List

from pypdf import PdfReader

from src.models import Page


def extract_pages(pdf_path: str) -> List[Page]:
    """Read a PDF and return its text, one Page per PDF page.

    Raises FileNotFoundError if the path doesn't exist, and RuntimeError
    if pypdf can't open the file at all (e.g. corrupted/encrypted PDF).
    Pages that extract as empty text (e.g. a scanned image page) are kept
    as empty strings rather than dropped, so page numbering stays aligned
    with the physical document.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF {pdf_path}: {exc}") from exc

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(Page(page_number=i, text=text))
    return pages


def extract_text(pdf_path: str) -> str:
    """Convenience helper: full document text as one string (pages joined
    with a blank line). Used for the Level 1 full-document fallback mode."""
    pages = extract_pages(pdf_path)
    return "\n\n".join(p.text for p in pages)
