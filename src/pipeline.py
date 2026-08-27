"""
Pipeline: extraction -> chunking -> indexing -> retrieval -> routing -> LLM.

load_document_index() does the one-time-per-document work. Per
question, answer_single_document() retrieves, classifies complexity,
routes to a model tier, calls the LLM, and runs the critique/revision
flow for reasoning-heavy (conflict/override) questions; other
questions pass through at one LLM call. answer_casual() skips
retrieval entirely for document-free conversation.
"""
from pathlib import Path
from typing import List, Optional, Tuple

from src import llm
from src.chunk import chunk_document
from src.critique import answer_with_critique
from src.extract import extract_pages
from src.index import EmbeddedIndex
from src.models import Answer, Chunk
from src.retrieve import DEFAULT_THRESHOLD, DEFAULT_TOP_K, decide
from src.route import COMPLEXITY_CONFLICT_OR_OVERRIDE, classify_complexity, detect_language, route


def make_doc_id(pdf_path: str) -> str:
    """Deterministic doc_id from a filename, e.g.
    'NON disclosure agreement Edited.pdf' -> 'non_disclosure_agreement_edited'.
    Same filename always produces the same id."""
    stem = Path(pdf_path).stem
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in stem)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def load_document_index(
    pdf_path: str, encode_fn=None
) -> Tuple[str, List[Chunk], EmbeddedIndex, str, str]:
    """Do the one-time-per-document work: extract, chunk, embed, and
    detect language.

    Returns (doc_id, chunks, index, full_document_text, language). Keep
    the returned index/full_text/language and pass them back into
    answer_single_document() for subsequent questions about this same
    document, so none of this is redone per question.
    """
    doc_id = make_doc_id(pdf_path)
    pages = extract_pages(pdf_path)
    chunks = chunk_document(pages, doc_id)
    index = EmbeddedIndex(chunks, encode_fn=encode_fn)
    full_text = "\n\n".join(p.text for p in pages)
    language = detect_language(full_text)
    return doc_id, chunks, index, full_text, language


def answer_single_document(
    question: str,
    pdf_path: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    doc_id: Optional[str] = None,
    index: Optional[EmbeddedIndex] = None,
    full_text: Optional[str] = None,
    language: Optional[str] = None,
    model: Optional[str] = None,
) -> Answer:
    """Answer one question against one document.

    Either pass `pdf_path` (and this function loads/embeds/detects
    language for you), or pass an already-built `doc_id`/`index`/
    `full_text`/`language` (from a prior load_document_index() call) to
    skip re-processing when asking multiple questions about the same
    document. If `language` is omitted while the other three are
    supplied, it's detected on the fly as a fallback so this function
    never crashes just because a caller forgot to pass it along.

    `model`, if given, bypasses routing and forces that exact model --
    useful for tests or a manual override. Normally leave it as None
    and let route.route() pick the tier.
    """
    if index is None or doc_id is None or full_text is None:
        if pdf_path is None:
            raise ValueError("Provide either pdf_path or doc_id+index+full_text.")
        doc_id, _chunks, index, full_text, language = load_document_index(pdf_path)
    elif language is None:
        language = detect_language(full_text)

    retrieval = decide(
        question=question,
        doc_id=doc_id,
        index=index,
        full_document_text=full_text,
        top_k=top_k,
        threshold=threshold,
    )

    complexity = classify_complexity(
        question, has_document=True, retrieval_mode=retrieval.mode
    )
    decision = route(
        language=language,
        complexity=complexity,
        doc_id=doc_id,
        question=question,
    )

    resolved_model = model or llm.resolve_model_for_tier(decision.tier)
    draft_answer = llm.call(question=question, context=retrieval.context, model=resolved_model)

    is_reasoning_heavy = complexity == COMPLEXITY_CONFLICT_OR_OVERRIDE
    critique_result = answer_with_critique(
        question=question,
        context=retrieval.context,
        draft_answer=draft_answer,
        is_reasoning_heavy=is_reasoning_heavy,
        reviewer_model=resolved_model,
        regenerate_model=resolved_model,
        doc_id=doc_id,
    )

    return Answer(
        question=question,
        answer=critique_result.final_answer,
        mode=retrieval.mode,
        confidence=retrieval.top_score,
        doc_id=doc_id,
        passages=retrieval.passages,
        model=resolved_model,
        tier=decision.tier,
        route_reason=decision.reason,
        n_llm_calls=critique_result.n_calls,
        critique_issue_found=critique_result.issue_found,
        critique_unresolved=critique_result.unresolved,
    )


def answer_casual(question: str, model: Optional[str] = None) -> Answer:
    """Answer a casual, document-independent question. Skips retrieval
    entirely since there is no document to search."""
    complexity = classify_complexity(question, has_document=False, retrieval_mode=None)
    decision = route(language="none", complexity=complexity, doc_id="", question=question)

    resolved_model = model or llm.resolve_model_for_tier(decision.tier)
    answer_text = llm.call(
        question=question,
        context="",
        model=resolved_model,
        system_prompt=llm.CASUAL_SYSTEM_PROMPT,
    )

    return Answer(
        question=question,
        answer=answer_text,
        mode="casual",
        confidence=0.0,
        doc_id="",
        passages=[],
        model=resolved_model,
        tier=decision.tier,
        route_reason=decision.reason,
        n_llm_calls=1,
    )
