"""
Legal-aware chunking.

Contracts already segment themselves via their own numbering ("1.",
"1.1", "ARTICLE II", "Section 4"). Splitting on those boundaries gives
chunks that line up with what a lawyer would call "a clause", which
makes them meaningful units to cite. If a document has little or no
detectable heading structure, falls back to fixed-size word windows so
nothing is ever left unchunked.
"""
import re
from typing import List

from src.models import Chunk, Page

# Matches a line that STARTS a numbered clause/section, e.g.:
#   "1. DEFINITIONS"
#   "1.1 Confidential Information shall mean..."
#   "2.3.4 Some deeply nested clause"
#   "ARTICLE II"  /  "Section 4"
HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+\S"
    r"|(?:ARTICLE|Article|SECTION|Section)\s+([IVXLCDM]+|\d+)\b)"
)

# Below this many heading-based chunks, we don't trust the structure
# enough to rely on it (e.g. a one-page letter with a single "1.").
MIN_HEADING_CHUNKS = 3

# Fixed-size fallback window, in words.
FALLBACK_CHUNK_WORDS = 200
FALLBACK_OVERLAP_WORDS = 30


def chunk_document(pages: List[Page], doc_id: str) -> List[Chunk]:
    """Split a document's pages into Chunks.

    Chunk IDs are deterministic: "{doc_id}::c000", "{doc_id}::c001", ...
    in document order, so re-chunking the same document always produces
    the same IDs.
    """
    chunks = _chunk_by_headings(pages, doc_id)
    if len(chunks) >= MIN_HEADING_CHUNKS:
        return chunks
    return _chunk_by_fixed_size(pages, doc_id)


def _chunk_by_headings(pages: List[Page], doc_id: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    current_lines: List[str] = []
    current_label = "preamble"
    current_page = pages[0].page_number if pages else 1
    next_index = 0

    def flush():
        nonlocal next_index
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::c{next_index:03d}",
                    section_label=current_label,
                    text=text,
                    page_number=current_page,
                )
            )
            next_index += 1

    for page in pages:
        for line in page.text.splitlines():
            if HEADING_RE.match(line):
                flush()
                current_lines = [line]
                current_label = line.strip()[:60] or "untitled clause"
                current_page = page.page_number
            else:
                current_lines.append(line)
    flush()
    return chunks


def _chunk_by_fixed_size(pages: List[Page], doc_id: str) -> List[Chunk]:
    # Flatten to (word, page_number) so each window can still report
    # which page it started on.
    words_with_pages = []
    for page in pages:
        for word in page.text.split():
            words_with_pages.append((word, page.page_number))

    chunks: List[Chunk] = []
    next_index = 0
    start = 0
    step = FALLBACK_CHUNK_WORDS - FALLBACK_OVERLAP_WORDS
    total = len(words_with_pages)

    while start < total:
        window = words_with_pages[start : start + FALLBACK_CHUNK_WORDS]
        text = " ".join(word for word, _ in window)
        page_number = window[0][1] if window else 1
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}::c{next_index:03d}",
                section_label=f"words {start}-{start + len(window)}",
                text=text,
                page_number=page_number,
            )
        )
        next_index += 1
        start += step

    return chunks
