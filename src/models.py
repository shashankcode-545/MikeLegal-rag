"""
Shared dataclasses used across the pipeline. Kept intentionally plain
(no inheritance, no validation frameworks) so they're easy to read and
reuse as-is.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Page:
    """One page of extracted PDF text."""
    page_number: int  # 1-indexed
    text: str


@dataclass
class Chunk:
    """A passage produced by chunk.py. `page_number` and `section_label`
    are carried through so later levels can cite back to a specific
    passage, not just a document."""
    doc_id: str
    chunk_id: str
    section_label: str
    text: str
    page_number: int


@dataclass
class RetrievalResult:
    """Output of retrieve.decide(). `context` is what actually gets sent
    to the LLM (either joined passages, or the full document)."""
    doc_id: str
    question: str
    mode: str  # "search" or "full_document"
    top_score: float
    n_passages: int
    top_k: int
    threshold: float
    context: str
    passages: List[Chunk] = field(default_factory=list)


@dataclass
class RouteDecision:
    """Output of route.route(). `language` and `complexity` are the two
    signals the routing rule used; `tier` is what it picked and
    `reason` is the short human-readable justification that gets
    logged alongside it."""
    doc_id: str
    question: str
    language: str  # "english" | "non_english" | "mixed" | "none"
    complexity: str  # "casual" | "simple_factual" | "full_document_reasoning" | "conflict_or_override" | "cross_document"
    tier: str  # "light" | "standard" | "strong"
    reason: str


@dataclass
class Answer:
    """Structured result returned by the pipeline."""
    question: str
    answer: str
    mode: str
    confidence: float
    doc_id: str
    passages: List[Chunk] = field(default_factory=list)
    model: Optional[str] = None
    tier: Optional[str] = None  # which model tier answered ("light"/"standard"/"strong")
    route_reason: Optional[str] = None  # why that tier was chosen
    n_llm_calls: Optional[int] = None  # total generation calls this question required
    critique_issue_found: Optional[bool] = None  # reviewer flagged a real issue
    critique_unresolved: Optional[bool] = None  # issue found AND regeneration couldn't fix it


@dataclass
class CrossDocExtraction:
    """One retrieved unit of evidence from a single document, used as
    input to the cross-document fusion prompt.

    `citation_tag` is exactly what the fusion answer is instructed to
    cite in square brackets -- either a real chunk_id (already precise
    to one passage, e.g. "some_doc::c003") or
    "{doc_id}::full_document" when that document's retrieval fell back
    to reading the whole thing. Either way, the tag alone is enough to
    trace a citation back to exactly what was retrieved, and back to
    the document it came from."""
    doc_id: str
    citation_tag: str
    text: str
    mode: str  # "search" or "full_document" -- the retrieval mode that produced it


@dataclass
class CrossDocAnswer:
    """Structured result returned by cross_doc.answer_cross_document().
    `citation_warnings` lists any citation in `answer` that could not
    be matched to a real `CrossDocExtraction` -- the request still
    succeeds; unverifiable citations are recorded, not silently
    dropped or treated as a failure."""
    question: str
    answer: str
    doc_ids: List[str] = field(default_factory=list)
    extractions: List[CrossDocExtraction] = field(default_factory=list)
    citation_warnings: List[str] = field(default_factory=list)
    tier: Optional[str] = None
    route_reason: Optional[str] = None
    n_llm_calls: Optional[int] = None
    critique_issue_found: Optional[bool] = None
    critique_unresolved: Optional[bool] = None
