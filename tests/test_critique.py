"""
Critique/revision tests (src/critique.py). LLM is mocked throughout.
"""
import json
from unittest.mock import patch

from src.critique import answer_with_critique


# Two clauses with a deliberately planted, unresolved conflict.
PLANTED_CONFLICT_CONTRACT = (
    "5. CONFIDENTIALITY. The Recipient's confidentiality obligations "
    "under this Section 5 shall survive termination of this Agreement "
    "indefinitely.\n\n"
    "9. TERMINATION. Upon termination or expiration of this Agreement "
    "for any reason, all obligations of both parties under this "
    "Agreement, including but not limited to Section 5, shall "
    "immediately terminate and be of no further force or effect."
)


def test_non_reasoning_heavy_question_never_triggers_review():
    """Definition of Done: a simple/conversational question must never
    trigger the review step -- exactly one call, no review."""
    with patch("src.critique.llm.call") as mock_call:
        result = answer_with_critique(
            question="What is the notice period?",
            context="Some ordinary contract text.",
            draft_answer="The notice period is thirty days.",
            is_reasoning_heavy=False,
            reviewer_model="fake/strong-model",
            regenerate_model="fake/strong-model",
        )

    assert mock_call.call_count == 0  # not reviewed at all
    assert result.n_calls == 1  # draft only (draft itself isn't made in this function)
    assert result.was_reviewed is False
    assert result.issue_found is False
    assert result.final_answer == "The notice period is thirty days."


def test_reasoning_heavy_question_with_no_issue_takes_two_calls():
    with patch("src.critique.llm.call") as mock_call:
        mock_call.return_value = "VERDICT: OK"

        result = answer_with_critique(
            question="Which clause governs if there's a conflict?",
            context=PLANTED_CONFLICT_CONTRACT,
            draft_answer="Section 5's confidentiality survives termination.",
            is_reasoning_heavy=True,
            reviewer_model="fake/strong-model",
            regenerate_model="fake/strong-model",
        )

    assert mock_call.call_count == 1  # review only, no regeneration needed
    assert result.n_calls == 2
    assert result.was_reviewed is True
    assert result.issue_found is False
    assert result.unresolved is False
    assert result.final_answer == "Section 5's confidentiality survives termination."


def test_planted_conflict_is_caught_and_resolved_by_regeneration():
    """Definition of Done: the review step must catch a deliberately
    planted, unresolved conflict and trigger a revised answer that
    resolves it."""
    flawed_draft = (
        "The confidentiality obligations in Section 5 survive termination, "
        "as stated in the agreement."
    )
    reviewer_verdict = (
        "VERDICT: ISSUE\n"
        "The draft ignores Section 9, which states that ALL obligations, "
        "explicitly including Section 5, terminate immediately upon "
        "termination. The draft never reconciles this direct contradiction."
    )
    resolved_answer = (
        "There is a direct conflict between Section 5 (confidentiality "
        "survives indefinitely) and Section 9 (all obligations, including "
        "Section 5, terminate immediately). The contract does not state "
        "which clause controls, so this is an unresolved drafting "
        "conflict that should be flagged to the parties rather than "
        "assumed either way."
    )

    with patch("src.critique.llm.call") as mock_call:
        mock_call.side_effect = [reviewer_verdict, resolved_answer]

        result = answer_with_critique(
            question="Does confidentiality survive termination of this agreement?",
            context=PLANTED_CONFLICT_CONTRACT,
            draft_answer=flawed_draft,
            is_reasoning_heavy=True,
            reviewer_model="fake/strong-model",
            regenerate_model="fake/strong-model",
        )

    assert mock_call.call_count == 2  # review + regenerate
    assert result.n_calls == 3
    assert result.was_reviewed is True
    assert result.issue_found is True
    assert result.unresolved is False
    assert result.final_answer == resolved_answer
    assert result.final_answer != flawed_draft  # the answer actually changed


def test_regeneration_that_cannot_resolve_falls_back_to_original_draft():
    """If the one allowed regeneration attempt still can't fix the
    issue, the original draft is sent back, with the discrepancy
    recorded -- not a fourth call, not the broken regenerated text."""
    original_draft = "Confidentiality survives termination under Section 5."
    reviewer_verdict = "VERDICT: ISSUE\nThis contradicts Section 9's blanket termination clause."
    failed_regeneration = (
        "UNRESOLVED: The contract text does not provide enough information "
        "to determine which clause controls in this conflict."
    )

    with patch("src.critique.llm.call") as mock_call:
        mock_call.side_effect = [reviewer_verdict, failed_regeneration]

        result = answer_with_critique(
            question="Does confidentiality survive termination?",
            context=PLANTED_CONFLICT_CONTRACT,
            draft_answer=original_draft,
            is_reasoning_heavy=True,
            reviewer_model="fake/strong-model",
            regenerate_model="fake/strong-model",
        )

    assert mock_call.call_count == 2  # review + one regeneration attempt, no more
    assert result.n_calls == 3
    assert result.issue_found is True
    assert result.unresolved is True
    assert result.final_answer == original_draft  # original sent back, not the failed attempt


def test_simple_question_takes_one_pass_reasoning_heavy_takes_more():
    """Definition of Done: visible evidence that a reasoning-heavy
    question takes more passes than a simple one."""
    with patch("src.critique.llm.call") as mock_call:
        simple_result = answer_with_critique(
            question="What is the notice period?",
            context="text",
            draft_answer="Thirty days.",
            is_reasoning_heavy=False,
            reviewer_model="m",
            regenerate_model="m",
        )
    assert simple_result.n_calls == 1

    with patch("src.critique.llm.call") as mock_call:
        mock_call.side_effect = ["VERDICT: ISSUE\nfeedback", "resolved answer"]
        heavy_result = answer_with_critique(
            question="conflicting clauses question",
            context=PLANTED_CONFLICT_CONTRACT,
            draft_answer="flawed draft",
            is_reasoning_heavy=True,
            reviewer_model="m",
            regenerate_model="m",
        )
    assert heavy_result.n_calls == 3
    assert heavy_result.n_calls > simple_result.n_calls


def test_critique_decision_is_logged(tmp_path):
    log_path = tmp_path / "log.jsonl"
    with patch("src.critique.llm.call") as mock_call:
        mock_call.side_effect = ["VERDICT: ISSUE\nfeedback here", "a fixed answer"]

        answer_with_critique(
            question="conflict question",
            context=PLANTED_CONFLICT_CONTRACT,
            draft_answer="flawed draft",
            is_reasoning_heavy=True,
            reviewer_model="m",
            regenerate_model="m",
            doc_id="doc1",
            log_path=log_path,
        )

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    required_fields = {
        "timestamp",
        "event",
        "doc_id",
        "question",
        "was_reviewed",
        "issue_found",
        "unresolved",
        "n_calls",
    }
    assert required_fields.issubset(entry.keys())
    assert entry["event"] == "critique"
    assert entry["n_calls"] == 3
    assert entry["issue_found"] is True
    assert entry["unresolved"] is False
