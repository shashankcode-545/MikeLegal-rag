"""
End-to-end pipeline tests (src/pipeline.py), with the LLM mocked.
"""
from unittest.mock import patch

from src.models import Answer
from src.pipeline import (
    answer_casual,
    answer_single_document,
    load_document_index,
    make_doc_id,
)
from tests.conftest import fake_encode


def test_make_doc_id_is_deterministic_and_slugified():
    doc_id_1 = make_doc_id("NON disclosure agreement Edited.pdf")
    doc_id_2 = make_doc_id("NON disclosure agreement Edited.pdf")

    assert doc_id_1 == doc_id_2
    assert " " not in doc_id_1
    assert doc_id_1 == doc_id_1.lower()


@patch("src.pipeline.llm.call")
def test_answer_single_document_returns_structured_answer(mock_llm_call, sample_pdf_path):
    mock_llm_call.return_value = "The notice period is thirty days, per Section 5."

    doc_id, chunks, index, full_text, language = load_document_index(
        sample_pdf_path, encode_fn=fake_encode
    )

    result = answer_single_document(
        question="What is the notice period for termination?",
        doc_id=doc_id,
        index=index,
        full_text=full_text,
        language=language,
        top_k=3,
        threshold=0.1,
    )

    assert isinstance(result, Answer)
    assert result.answer == "The notice period is thirty days, per Section 5."
    assert result.mode in {"search", "full_document"}
    assert result.doc_id == doc_id
    assert result.tier in {"light", "standard", "strong"}  # Level 2: routing always sets a tier
    assert mock_llm_call.call_count == 1  # exactly one generation call


@patch("src.pipeline.llm.call")
def test_index_is_reused_across_multiple_questions(mock_llm_call, sample_pdf_path):
    """Confirms load_document_index() only needs to run once, and the
    same index/full_text can answer multiple questions."""
    mock_llm_call.return_value = "mocked answer"

    doc_id, chunks, index, full_text, language = load_document_index(
        sample_pdf_path, encode_fn=fake_encode
    )

    result_1 = answer_single_document(
        "What is the notice period?",
        doc_id=doc_id,
        index=index,
        full_text=full_text,
        language=language,
    )
    result_2 = answer_single_document(
        "Who are the parties?",
        doc_id=doc_id,
        index=index,
        full_text=full_text,
        language=language,
    )

    assert result_1.doc_id == result_2.doc_id == doc_id
    assert mock_llm_call.call_count == 2  # one call per question, index built only once


# --- Routing wired into the pipeline ---

@patch("src.pipeline.llm.call")
def test_simple_factual_question_routes_to_light_tier(mock_llm_call, sample_pdf_path):
    """A narrow factual question against an English document should end
    up on the light tier (assuming it gets a confident search-mode hit)."""
    mock_llm_call.return_value = "mocked answer"

    doc_id, chunks, index, full_text, language = load_document_index(
        sample_pdf_path, encode_fn=fake_encode
    )
    assert language == "english"

    result = answer_single_document(
        "What is the notice period for termination?",
        doc_id=doc_id,
        index=index,
        full_text=full_text,
        language=language,
        top_k=3,
        threshold=0.1,  # low enough to force a confident search-mode hit
    )

    assert result.mode == "search"
    assert result.tier == "light"
    assert result.route_reason is not None


@patch("src.pipeline.llm.call")
def test_full_document_fallback_routes_to_standard_tier(mock_llm_call, sample_pdf_path):
    """When Level 1 falls back to full-document mode, Level 2 should
    treat that as full-document reasoning and route to the standard tier."""
    mock_llm_call.return_value = "mocked answer"

    doc_id, chunks, index, full_text, language = load_document_index(
        sample_pdf_path, encode_fn=fake_encode
    )

    result = answer_single_document(
        "some obscure question unlikely to match any clause",
        doc_id=doc_id,
        index=index,
        full_text=full_text,
        language=language,
        threshold=0.999,  # force fallback
    )

    assert result.mode == "full_document"
    assert result.tier == "standard"


@patch("src.pipeline.llm.call")
def test_conflict_question_routes_to_strong_tier(mock_llm_call, sample_pdf_path):
    mock_llm_call.return_value = "mocked answer"

    doc_id, chunks, index, full_text, language = load_document_index(
        sample_pdf_path, encode_fn=fake_encode
    )

    result = answer_single_document(
        "Which clause takes precedence if there is a conflict between sections?",
        doc_id=doc_id,
        index=index,
        full_text=full_text,
        language=language,
        threshold=0.1,
    )

    assert result.tier == "strong"


@patch("src.pipeline.llm.call")
def test_non_english_document_always_routes_to_strong_tier(mock_llm_call):
    """Even a simple factual question over a non-English document should
    route to the strong tier -- language overrides complexity."""
    mock_llm_call.return_value = "mocked answer"

    from src.models import Chunk
    from src.index import EmbeddedIndex

    chunks = [
        Chunk(
            doc_id="fr_doc",
            chunk_id="fr_doc::c000",
            section_label="1. Termination",
            text="Chaque partie peut resilier cet accord avec un preavis de trente jours.",
            page_number=1,
        )
    ]
    index = EmbeddedIndex(chunks, encode_fn=fake_encode)
    full_text = chunks[0].text

    result = answer_single_document(
        "What is the notice period?",
        doc_id="fr_doc",
        index=index,
        full_text=full_text,
        language="non_english",  # as detect_language would classify this document
        threshold=0.1,
    )

    assert result.tier == "strong"


@patch("src.pipeline.llm.call")
def test_answer_casual_skips_retrieval_and_routes_light(mock_llm_call):
    mock_llm_call.return_value = "Hey! How can I help?"

    result = answer_casual("hi there")

    assert isinstance(result, Answer)
    assert result.mode == "casual"
    assert result.tier == "light"
    assert result.passages == []
    assert result.doc_id == ""
    mock_llm_call.assert_called_once()
    _, kwargs = mock_llm_call.call_args
    assert kwargs["system_prompt"] == llm_module_casual_prompt()


def llm_module_casual_prompt():
    from src import llm

    return llm.CASUAL_SYSTEM_PROMPT


@patch("src.pipeline.llm.call")
def test_explicit_model_override_bypasses_routing(mock_llm_call, sample_pdf_path):
    """Passing model= explicitly should still work (e.g. for tests or a
    manual override) without breaking the routing decision that gets logged."""
    mock_llm_call.return_value = "mocked answer"

    doc_id, chunks, index, full_text, language = load_document_index(
        sample_pdf_path, encode_fn=fake_encode
    )

    result = answer_single_document(
        "What is the notice period?",
        doc_id=doc_id,
        index=index,
        full_text=full_text,
        language=language,
        model="some/explicit-model",
        threshold=0.1,
    )

    assert result.model == "some/explicit-model"
    assert result.tier in {"light", "standard", "strong"}  # routing still ran and was logged


@patch("src.pipeline.llm.call")
@patch("src.pipeline.detect_language")
def test_language_is_detected_once_and_cached_across_questions(
    mock_detect_language, mock_llm_call, sample_pdf_path
):
    """Language detection must run once per document (inside
    load_document_index), not once per question."""
    mock_detect_language.return_value = "english"
    mock_llm_call.return_value = "mocked answer"

    doc_id, chunks, index, full_text, language = load_document_index(
        sample_pdf_path, encode_fn=fake_encode
    )
    assert mock_detect_language.call_count == 1  # called during loading, once

    answer_single_document(
        "What is the notice period?",
        doc_id=doc_id,
        index=index,
        full_text=full_text,
        language=language,  # caller passes the cached value along
    )
    answer_single_document(
        "Who are the parties?",
        doc_id=doc_id,
        index=index,
        full_text=full_text,
        language=language,
    )

    assert mock_detect_language.call_count == 1


@patch("src.pipeline.llm.call")
def test_routed_model_is_actually_passed_to_llm_call(mock_llm_call, sample_pdf_path):
    """Confirms the tier's resolved model id is the exact `model=` value
    llm.call() receives -- not just that Answer.model looks right."""
    mock_llm_call.return_value = "mocked answer"

    doc_id, chunks, index, full_text, language = load_document_index(
        sample_pdf_path, encode_fn=fake_encode
    )

    result = answer_single_document(
        "What is the notice period for termination?",
        doc_id=doc_id,
        index=index,
        full_text=full_text,
        language=language,
        threshold=0.1,
    )

    from src import llm

    expected_model = llm.resolve_model_for_tier(result.tier)
    _, kwargs = mock_llm_call.call_args
    assert kwargs["model"] == expected_model
    assert result.model == expected_model


# --- Critique/revision wired into the pipeline ---

@patch("src.llm.call")
def test_simple_question_never_triggers_critique_in_pipeline(mock_llm_call, sample_pdf_path):
    """Definition of Done: a simple/conversational question never
    triggers the review step, end to end through the real pipeline."""
    mock_llm_call.return_value = "The notice period is thirty days."

    doc_id, chunks, index, full_text, language = load_document_index(
        sample_pdf_path, encode_fn=fake_encode
    )

    result = answer_single_document(
        "What is the notice period for termination?",
        doc_id=doc_id,
        index=index,
        full_text=full_text,
        language=language,
        threshold=0.1,
    )

    assert result.tier == "light"
    assert result.n_llm_calls == 1
    assert result.critique_issue_found is False
    mock_llm_call.assert_called_once()  # the one draft call -- critique never ran


@patch("src.llm.call")
def test_conflict_question_triggers_review_in_pipeline(mock_llm_call, sample_pdf_path):
    """A conflict/override question routes to the strong tier (Level 2)
    and then goes through critique (Level 3): draft + review, with
    review saying OK -> 2 total calls."""
    mock_llm_call.side_effect = [
        "draft answer about the conflict",  # 1: the draft
        "VERDICT: OK",  # 2: the review verdict
    ]

    doc_id, chunks, index, full_text, language = load_document_index(
        sample_pdf_path, encode_fn=fake_encode
    )

    result = answer_single_document(
        "Which clause takes precedence if there is a conflict between sections?",
        doc_id=doc_id,
        index=index,
        full_text=full_text,
        language=language,
        threshold=0.1,
    )

    assert result.tier == "strong"
    assert result.n_llm_calls == 2  # draft + review, review said OK
    assert result.critique_issue_found is False
    assert result.answer == "draft answer about the conflict"  # draft stands, unchanged
    assert mock_llm_call.call_count == 2


@patch("src.llm.call")
def test_conflict_question_with_flagged_issue_triggers_regeneration_in_pipeline(
    mock_llm_call, sample_pdf_path
):
    """Full 3-call path through the real pipeline: draft, review (flags
    an issue), regenerate -- and the final answer is the regenerated
    text, not the flawed draft."""
    mock_llm_call.side_effect = [
        "flawed draft answer",  # 1: the draft
        "VERDICT: ISSUE\nThe draft ignores a conflicting clause.",  # 2: review
        "a corrected answer that reconciles the conflict",  # 3: regenerate
    ]

    doc_id, chunks, index, full_text, language = load_document_index(
        sample_pdf_path, encode_fn=fake_encode
    )

    result = answer_single_document(
        "Which clause overrides the other in case of conflict?",
        doc_id=doc_id,
        index=index,
        full_text=full_text,
        language=language,
        threshold=0.1,
    )

    assert result.n_llm_calls == 3
    assert result.critique_issue_found is True
    assert result.critique_unresolved is False
    assert result.answer == "a corrected answer that reconciles the conflict"
    assert mock_llm_call.call_count == 3
