"""
Cross-document citation grounding.

For questions spanning more than one document: retrieve evidence from
each document separately (reusing retrieve.decide(), no LLM calls),
fuse it into one prompt tagged with per-passage citation tags, make a
single LLM call routed to the strongest tier and through the critique
flow, then verify every citation the answer makes against what was
actually retrieved. Total LLM calls: 1-3, independent of document
count, since only the single fused answer is generated/reviewed.
"""
import re
from typing import List, NamedTuple, Optional

from src import llm
from src.critique import answer_with_critique
from src.index import EmbeddedIndex
from src.models import CrossDocAnswer, CrossDocExtraction
from src.pipeline import load_document_index
from src.retrieve import DEFAULT_THRESHOLD, DEFAULT_TOP_K, decide
from src.route import COMPLEXITY_CROSS_DOCUMENT, route

CROSS_DOC_SYSTEM_PROMPT = (
    "You are a legal document assistant comparing information across "
    "multiple contracts. You will be given several excerpts, each "
    "labeled with a citation tag in square brackets, e.g. "
    "[some_doc::c003]. Answer the question using ONLY the excerpts "
    "given below. For every claim, cite the exact tag(s) it comes from "
    "in square brackets. Do NOT invent a citation tag, document, or "
    "clause that is not shown below -- if the excerpts do not contain "
    "enough information to answer, say so explicitly instead of "
    "guessing or citing something that wasn't given to you."
)


class LoadedDocument(NamedTuple):
    """A convenience wrapper around load_document_index()'s return
    value, with named fields instead of tuple positions. Purely a
    readability aid local to this module -- pipeline.py's own return
    shape is untouched."""
    doc_id: str
    index: EmbeddedIndex
    full_text: str
    language: str


def load_documents(pdf_paths: List[str], encode_fn=None) -> List[LoadedDocument]:
    """Load and embed each document ONCE, so answer_cross_document() can
    be called repeatedly across many questions without re-extracting,
    re-chunking, or re-embedding any document."""
    loaded = []
    for path in pdf_paths:
        doc_id, _chunks, index, full_text, language = load_document_index(
            path, encode_fn=encode_fn
        )
        loaded.append(
            LoadedDocument(doc_id=doc_id, index=index, full_text=full_text, language=language)
        )
    return loaded


def extract_from_documents(
    question: str,
    documents: List[LoadedDocument],
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[CrossDocExtraction]:
    """Run retrieve.decide() once per document. A search-mode hit
    produces one CrossDocExtraction per retrieved chunk, citable by
    its chunk_id; a full-document fallback produces one extraction
    citable as "{doc_id}::full_document"."""
    extractions: List[CrossDocExtraction] = []
    for doc in documents:
        result = decide(
            question=question,
            doc_id=doc.doc_id,
            index=doc.index,
            full_document_text=doc.full_text,
            top_k=top_k,
            threshold=threshold,
        )
        if result.mode == "search" and result.passages:
            for chunk in result.passages:
                extractions.append(
                    CrossDocExtraction(
                        doc_id=doc.doc_id,
                        citation_tag=chunk.chunk_id,
                        text=chunk.text,
                        mode="search",
                    )
                )
        else:
            extractions.append(
                CrossDocExtraction(
                    doc_id=doc.doc_id,
                    citation_tag=f"{doc.doc_id}::full_document",
                    text=doc.full_text,
                    mode="full_document",
                )
            )
    return extractions


def build_fusion_prompt(question: str, extractions: List[CrossDocExtraction]) -> str:
    """Build the single prompt handed to the fusion LLM call: every
    excerpt, tagged with its citation tag, followed by the question."""
    blocks = [
        f"[{extraction.citation_tag}] (from document '{extraction.doc_id}'):\n{extraction.text}"
        for extraction in extractions
    ]
    excerpts_text = "\n\n".join(blocks)
    return f"Excerpts:\n{excerpts_text}\n\nQuestion: {question}"


_CITATION_TAG_RE = re.compile(r"\[([^\]]+)\]")


def verify_citations(answer_text: str, extractions: List[CrossDocExtraction]) -> List[str]:
    """Check every bracketed citation tag in `answer_text` against the
    tags that were actually retrieved. Returns a list of warning
    strings -- one per distinct unverifiable tag. Never raises and
    never modifies `answer_text`: an unverifiable citation is surfaced
    as a warning, not a failure.
    """
    known_tags = {extraction.citation_tag for extraction in extractions}
    warnings: List[str] = []
    seen_bad_tags = set()
    for tag in _CITATION_TAG_RE.findall(answer_text):
        if tag not in known_tags and tag not in seen_bad_tags:
            warnings.append(f"Citation '[{tag}]' could not be verified against retrieved passages.")
            seen_bad_tags.add(tag)
    return warnings


def _combine_languages(languages: List[str]) -> str:
    """Combine per-document language buckets into one signal for
    routing. If ANY document involved is non-English or mixed, the
    whole cross-document question is treated as needing the strongest
    tier for language safety."""
    if any(language in ("non_english", "mixed") for language in languages):
        return "mixed"
    return "english"


def answer_cross_document(
    question: str,
    documents: List[LoadedDocument],
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    model: Optional[str] = None,
) -> CrossDocAnswer:
    """Answer one question spanning multiple documents.

    `documents` should come from load_documents() so this can be
    called repeatedly across many questions without re-embedding
    anything. Makes one fusion LLM call, then always routes it through
    the critique flow (cross-document comparison is treated as
    reasoning-heavy), adding 0-2 more calls.
    """
    if not documents:
        raise ValueError("answer_cross_document requires at least one loaded document.")

    extractions = extract_from_documents(question, documents, top_k=top_k, threshold=threshold)

    combined_language = _combine_languages([doc.language for doc in documents])
    combined_doc_id = ",".join(doc.doc_id for doc in documents)
    decision = route(
        language=combined_language,
        complexity=COMPLEXITY_CROSS_DOCUMENT,
        doc_id=combined_doc_id,
        question=question,
    )

    resolved_model = model or llm.resolve_model_for_tier(decision.tier)

    fusion_prompt = build_fusion_prompt(question, extractions)
    draft_answer = llm.call(
        question=fusion_prompt,
        context="",
        model=resolved_model,
        system_prompt=CROSS_DOC_SYSTEM_PROMPT,
    )

    # Give the reviewer the same excerpts the fusion step saw, so it can
    # check the draft's claims and citations against the real evidence.
    critique_context = "\n\n".join(
        f"[{extraction.citation_tag}] {extraction.text}" for extraction in extractions
    )
    critique_result = answer_with_critique(
        question=question,
        context=critique_context,
        draft_answer=draft_answer,
        is_reasoning_heavy=True,
        reviewer_model=resolved_model,
        regenerate_model=resolved_model,
        doc_id=combined_doc_id,
    )

    citation_warnings = verify_citations(critique_result.final_answer, extractions)

    return CrossDocAnswer(
        question=question,
        answer=critique_result.final_answer,
        doc_ids=[doc.doc_id for doc in documents],
        extractions=extractions,
        citation_warnings=citation_warnings,
        tier=decision.tier,
        route_reason=decision.reason,
        n_llm_calls=critique_result.n_calls,
        critique_issue_found=critique_result.issue_found,
        critique_unresolved=critique_result.unresolved,
    )
