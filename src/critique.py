"""
Critique and revision for reasoning-heavy questions.

Adds a distinct "reviewer" pass over a draft answer and, only if the
reviewer flags a real issue, one regeneration attempt with that
feedback folded in. Costs 1 call if not reasoning-heavy, 2 if reviewed
and OK, 3 if reviewed, flagged, and regenerated -- never more than
that, since only one regeneration attempt is ever made. The regenerate
prompt asks the model to prefix "UNRESOLVED:" if it still can't fix
the issue, so a single call's text tells us fixed-vs-not without a
fourth call.
"""
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src import llm

DEFAULT_LOG_PATH = Path("logs/run_log.jsonl")

REVIEWER_SYSTEM_PROMPT = (
    "You are an independent legal reviewer. You did NOT write the draft "
    "answer below -- your job is only to check it against the contract "
    "text for three specific problems:\n"
    "1. A carve-out, exception, or override the draft claims to have "
    "addressed but did not actually resolve.\n"
    "2. Any claim in the draft that is not actually supported by the "
    "contract text.\n"
    "3. A contradiction between two clauses that the draft failed to "
    "reconcile.\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: OK\n"
    "or\n"
    "VERDICT: ISSUE\n"
    "<one short paragraph explaining exactly what is wrong and what "
    "the answer needs to address>"
)

REGENERATE_SYSTEM_PROMPT = (
    "You are a legal document assistant revising a previous answer "
    "based on independent reviewer feedback. Using ONLY the contract "
    "text supplied, produce a corrected answer that addresses the "
    "feedback. If, after reviewing the contract text again, you "
    "genuinely cannot resolve the reviewer's concern with the "
    "information available, start your entire response with "
    "'UNRESOLVED:' followed by a short explanation instead of guessing."
)


@dataclass
class CritiqueResult:
    """What answer_with_critique() reports back: the final answer text
    plus bookkeeping the pipeline needs for logging."""
    final_answer: str
    n_calls: int
    was_reviewed: bool
    issue_found: bool  # the reviewer flagged "VERDICT: ISSUE"
    unresolved: bool  # issue_found=True AND the regeneration couldn't fix it
    review_verdict: Optional[str] = None  # raw reviewer text, for logging/debugging


def review(question: str, draft_answer: str, source_text: str, model: str) -> str:
    """One reviewer call. Returns the raw verdict text, starting with
    either 'VERDICT: OK' or 'VERDICT: ISSUE ...'."""
    reviewer_prompt = (
        f"Question: {question}\n\n"
        f"Draft answer to review:\n{draft_answer}\n\n"
        f"Contract text:\n{source_text}"
    )
    return llm.call(
        question=reviewer_prompt,
        context="",  # everything needed is already folded into the prompt above
        model=model,
        system_prompt=REVIEWER_SYSTEM_PROMPT,
    )


def regenerate(question: str, context: str, feedback: str, model: str) -> str:
    """One regeneration call, folding the reviewer's feedback in. The
    model is instructed to prefix 'UNRESOLVED:' if it still can't fix
    the issue, so the caller can tell fixed-vs-not from this one call's
    text alone."""
    prompt = (
        f"Original question: {question}\n\n"
        f"Independent reviewer's feedback on your previous answer:\n{feedback}\n\n"
        f"Contract context:\n{context}\n\n"
        "Provide a corrected answer that addresses this feedback."
    )
    return llm.call(
        question=prompt,
        context="",
        model=model,
        system_prompt=REGENERATE_SYSTEM_PROMPT,
    )


def _verdict_is_issue(verdict_text: str) -> bool:
    return bool(re.search(r"VERDICT:\s*ISSUE", verdict_text, re.IGNORECASE))


def _extract_feedback(verdict_text: str) -> str:
    """Everything after the VERDICT line is the feedback to hand to
    the regeneration step."""
    return re.sub(
        r"VERDICT:\s*(OK|ISSUE)\s*", "", verdict_text, flags=re.IGNORECASE
    ).strip()


def _is_unresolved(regenerated_text: str) -> bool:
    return regenerated_text.strip().upper().startswith("UNRESOLVED")


def answer_with_critique(
    question: str,
    context: str,
    draft_answer: str,
    is_reasoning_heavy: bool,
    reviewer_model: str,
    regenerate_model: str,
    doc_id: str = "",
    log_path: Optional[Path] = None,
) -> CritiqueResult:
    """Run the critique/revision flow on an already-generated draft, if
    (and only if) `is_reasoning_heavy` is True. `draft_answer` is
    passed in rather than generated here, so this only ever adds calls
    on top of the caller's draft call, never duplicates it."""
    if not is_reasoning_heavy:
        result = CritiqueResult(
            final_answer=draft_answer,
            n_calls=1,
            was_reviewed=False,
            issue_found=False,
            unresolved=False,
        )
        _log_critique_decision(doc_id, question, result, log_path)
        return result

    verdict_text = review(question, draft_answer, context, model=reviewer_model)

    if not _verdict_is_issue(verdict_text):
        result = CritiqueResult(
            final_answer=draft_answer,
            n_calls=2,
            was_reviewed=True,
            issue_found=False,
            unresolved=False,
            review_verdict=verdict_text,
        )
        _log_critique_decision(doc_id, question, result, log_path)
        return result

    feedback = _extract_feedback(verdict_text)
    revised_answer = regenerate(question, context, feedback, model=regenerate_model)

    if _is_unresolved(revised_answer):
        # One round was allowed and it didn't fix the issue -- ship the
        # original draft, but make the discrepancy visible.
        result = CritiqueResult(
            final_answer=draft_answer,
            n_calls=3,
            was_reviewed=True,
            issue_found=True,
            unresolved=True,
            review_verdict=verdict_text,
        )
    else:
        result = CritiqueResult(
            final_answer=revised_answer,
            n_calls=3,
            was_reviewed=True,
            issue_found=True,
            unresolved=False,
            review_verdict=verdict_text,
        )

    _log_critique_decision(doc_id, question, result, log_path)
    return result


def _log_critique_decision(
    doc_id: str, question: str, result: CritiqueResult, log_path: Optional[Path]
) -> None:
    """Append one JSON line per question, including the 1-call,
    never-reviewed case, so the log shows call counts for every
    question, not just the expensive ones."""
    resolved_path = log_path if log_path is not None else DEFAULT_LOG_PATH
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "event": "critique",
        "doc_id": doc_id,
        "question": question,
        "was_reviewed": result.was_reviewed,
        "issue_found": result.issue_found,
        "unresolved": result.unresolved,
        "n_calls": result.n_calls,
    }
    with open(resolved_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
