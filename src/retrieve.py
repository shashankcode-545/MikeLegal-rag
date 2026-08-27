"""
The retrieval-vs-full-read decision.

This is a key quality signal the rest of the pipeline depends on. It
is intentionally a single, pure-ish function: given a question and an
already-built index, decide whether a confident enough
passage match exists ("search" mode) or whether the system should fall
back to reading the entire document ("full_document" mode). Routing
reuses this function's output directly as one of its inputs -- so it
must stay simple and stable rather than growing extra responsibilities.
"""
import json
import time
from pathlib import Path
from typing import Optional

from src.index import EmbeddedIndex
from src.models import RetrievalResult

DEFAULT_TOP_K = 3
DEFAULT_THRESHOLD = 0.35

DEFAULT_LOG_PATH = Path("logs/run_log.jsonl")


def decide(
    question: str,
    doc_id: str,
    index: EmbeddedIndex,
    full_document_text: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    log_path: Optional[Path] = None,
) -> RetrievalResult:
    """Decide whether to answer from retrieved passages or the full
    document, and log that decision.

    `top_k` and `threshold` are plain function parameters -- this is
    what makes retrieval sensitivity "configurable per request" without
    any config file, server flag, or UI: callers just pass different
    values, and the defaults above are what's used if they don't.
    """
    hits = index.search(question, top_k=top_k)
    top_score = hits[0][1] if hits else 0.0

    if hits and top_score >= threshold:
        mode = "search"
        passages = [chunk for chunk, _score in hits]
        context = "\n\n".join(f"[{c.section_label}] {c.text}" for c in passages)
    else:
        mode = "full_document"
        passages = []
        context = full_document_text

    result = RetrievalResult(
        doc_id=doc_id,
        question=question,
        mode=mode,
        top_score=top_score,
        n_passages=len(passages),
        top_k=top_k,
        threshold=threshold,
        context=context,
        passages=passages,
    )

    _log_decision(result, log_path if log_path is not None else DEFAULT_LOG_PATH)
    return result


def _log_decision(result: RetrievalResult, log_path: Path) -> None:
    """Append one JSON line per decision. Intentionally logs only
    summary fields (not full chunk text) to keep the log small and
    readable -- passage text is still available on the returned
    RetrievalResult for anything that needs it in-process."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "doc_id": result.doc_id,
        "question": result.question,
        "mode": result.mode,
        "top_score": result.top_score,
        "n_passages": result.n_passages,
        "top_k": result.top_k,
        "threshold": result.threshold,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
