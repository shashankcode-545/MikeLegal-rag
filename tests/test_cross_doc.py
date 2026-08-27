"""
Level 4 tests: cross-document citation grounding (src/cross_doc.py).

No real OpenRouter calls -- llm.call is mocked throughout.
"""
import json
from unittest.mock import patch

import pytest

from src.cross_doc import (
    LoadedDocument,
    answer_cross_document,
    build_fusion_prompt,
    extract_from_documents,
    load_documents,
    verify_citations,
)
from src.index import EmbeddedIndex
from src.models import Chunk
from tests.conftest import fake_encode


def make_document(doc_id: str, clauses: dict, language: str = "english") -> LoadedDocument:
    """Build a synthetic LoadedDocument from {section_label: text} clauses."""
    chunks = [
        Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}::c{i:03d}",
            section_label=label,
            text=text,
            page_number=1,
        )
        for i, (label, text) in enumerate(clauses.items())
    ]
    index = EmbeddedIndex(chunks, encode_fn=fake_encode)
    full_text = " ".join(clauses.values())
    return LoadedDocument(doc_id=doc_id, index=index, full_text=full_text, language=language)


NDA_DOC = make_document(
    "nda_doc",
    {
        "1. Confidentiality": "The Recipient shall keep confidential information secret for five years.",
        "2. Termination": "Either party may terminate this agreement with thirty days written notice.",
    },
)

SERVICE_DOC = make_document(
    "service_doc",
    {
        "1. Confidentiality": "Confidential information disclosed under this agreement must be protected for two years.",
        "2. Payment Terms": "Payment is due within forty-five days of invoice.",
    },
)


# --- Per-document extraction ---

def test_extract_from_documents_returns_one_or_more_per_document():
    extractions = extract_from_documents(
        "How long must confidential information be kept secret?",
        [NDA_DOC, SERVICE_DOC],
        top_k=1,
        threshold=0.05,
    )

    doc_ids_seen = {e.doc_id for e in extractions}
    assert doc_ids_seen == {"nda_doc", "service_doc"}  # evidence pulled from BOTH documents


def test_extraction_citation_tags_trace_back_to_real_chunks():
    extractions = extract_from_documents(
        "How long must confidential information be kept secret?",
        [NDA_DOC, SERVICE_DOC],
        top_k=1,
        threshold=0.05,
    )
    for extraction in extractions:
        if extraction.mode == "search":
            # citation_tag must be a real chunk_id, precise to one passage
            assert extraction.citation_tag.startswith(extraction.doc_id + "::c")


def test_extraction_falls_back_to_full_document_tag_when_no_confident_match():
    extractions = extract_from_documents(
        "some completely unrelated question about zoning permits",
        [NDA_DOC],
        top_k=1,
        threshold=0.999,  # unreachable -> forces Level 1's full-document fallback
    )
    assert len(extractions) == 1
    assert extractions[0].mode == "full_document"
    assert extractions[0].citation_tag == "nda_doc::full_document"


# --- Fusion prompt ---

def test_fusion_prompt_includes_every_extraction_tag():
    extractions = extract_from_documents(
        "How long must confidential information be kept secret?",
        [NDA_DOC, SERVICE_DOC],
        top_k=1,
        threshold=0.05,
    )
    prompt = build_fusion_prompt("How long must confidentiality last?", extractions)

    for extraction in extractions:
        assert f"[{extraction.citation_tag}]" in prompt
    assert "How long must confidentiality last?" in prompt


# --- Citation verification ---

def test_verify_citations_all_traceable_produces_no_warnings():
    extractions = extract_from_documents(
        "How long must confidential information be kept secret?",
        [NDA_DOC, SERVICE_DOC],
        top_k=1,
        threshold=0.05,
    )
    real_tag_1 = extractions[0].citation_tag
    real_tag_2 = extractions[1].citation_tag
    answer_text = (
        f"The NDA requires five years [{real_tag_1}], while the service "
        f"agreement requires two years [{real_tag_2}]."
    )

    warnings = verify_citations(answer_text, extractions)
    assert warnings == []


def test_verify_citations_flags_a_deliberately_unverifiable_citation():
    extractions = extract_from_documents(
        "How long must confidential information be kept secret?",
        [NDA_DOC, SERVICE_DOC],
        top_k=1,
        threshold=0.05,
    )
    real_tag = extractions[0].citation_tag
    fabricated_tag = "some_other_contract::c999"  # never retrieved, not in the collection
    answer_text = (
        f"The NDA requires five years [{real_tag}], and per "
        f"[{fabricated_tag}] there is also a ten-year exception."
    )

    warnings = verify_citations(answer_text, extractions)
    assert len(warnings) == 1
    assert fabricated_tag in warnings[0]


def test_verify_citations_never_raises_and_does_not_fail_the_request():
    """An unverifiable citation is a warning, not an exception."""
    extractions = extract_from_documents(
        "How long must confidential information be kept secret?", [NDA_DOC], top_k=1, threshold=0.05
    )
    answer_text = "This cites [a_completely_made_up_document::c001] which was never retrieved."

    warnings = verify_citations(answer_text, extractions)  # must not raise
    assert len(warnings) == 1


def test_verify_citations_deduplicates_repeated_bad_tags():
    extractions = extract_from_documents(
        "How long must confidential information be kept secret?", [NDA_DOC], top_k=1, threshold=0.05
    )
    fabricated_tag = "ghost_doc::c001"
    answer_text = f"See [{fabricated_tag}] and again [{fabricated_tag}] for confirmation."

    warnings = verify_citations(answer_text, extractions)
    assert len(warnings) == 1  # one warning per distinct bad tag, not per occurrence


# --- End-to-end cross-document answer (mocked LLM) ---

@patch("src.llm.call")
def test_answer_cross_document_combined_answer_with_traceable_citations(mock_llm_call):
    """Definition of Done: a combined, multi-document answer where
    citations are checked against what was retrieved -- the
    fully-traceable case."""
    extractions_preview = extract_from_documents(
        "How long must confidentiality be kept?", [NDA_DOC, SERVICE_DOC], top_k=1, threshold=0.05
    )
    tag_1, tag_2 = extractions_preview[0].citation_tag, extractions_preview[1].citation_tag

    mock_llm_call.side_effect = [
        f"The NDA requires five years [{tag_1}], while the service agreement requires two years [{tag_2}].",
        "VERDICT: OK",
    ]

    result = answer_cross_document(
        "How long must confidentiality be kept?",
        [NDA_DOC, SERVICE_DOC],
        top_k=1,
        threshold=0.05,
    )

    assert set(result.doc_ids) == {"nda_doc", "service_doc"}
    assert result.citation_warnings == []  # every citation traceable
    assert result.tier == "strong"  # cross-document always routes to strong
    assert result.n_llm_calls == 2  # fusion + review (review said OK)


@patch("src.llm.call")
def test_answer_cross_document_flags_unverifiable_citation_without_failing(mock_llm_call):
    """Definition of Done: the deliberately unverifiable citation case
    -- the request still succeeds, the bad citation is recorded as a
    warning."""
    mock_llm_call.side_effect = [
        "According to [totally_fabricated_doc::c007], confidentiality lasts forever.",
        "VERDICT: OK",
    ]

    result = answer_cross_document(
        "How long must confidentiality be kept?",
        [NDA_DOC, SERVICE_DOC],
        top_k=1,
        threshold=0.05,
    )

    assert len(result.citation_warnings) == 1
    assert "totally_fabricated_doc::c007" in result.citation_warnings[0]
    assert result.answer  # the answer is still returned, not dropped/failed


@patch("src.llm.call")
def test_answer_cross_document_uses_exactly_one_fusion_call_when_not_reviewed_further(
    mock_llm_call,
):
    mock_llm_call.side_effect = ["a combined answer", "VERDICT: OK"]
    result = answer_cross_document("compare payment terms", [NDA_DOC, SERVICE_DOC], threshold=0.05)
    assert mock_llm_call.call_count == 2  # fusion + review; no regeneration needed
    assert result.n_llm_calls == 2


@patch("src.llm.call")
def test_answer_cross_document_call_count_independent_of_document_count(mock_llm_call):
    """LLM call count must not scale with the number of documents compared."""
    third_doc = make_document(
        "lease_doc",
        {"1. Confidentiality": "Tenant information shall remain confidential for one year."},
    )
    mock_llm_call.side_effect = ["a combined answer across three docs", "VERDICT: OK"]

    result = answer_cross_document(
        "How long must confidentiality be kept across these agreements?",
        [NDA_DOC, SERVICE_DOC, third_doc],
        threshold=0.05,
    )

    assert set(result.doc_ids) == {"nda_doc", "service_doc", "lease_doc"}
    assert mock_llm_call.call_count == 2  # still just fusion + review, not 3x anything


@patch("src.llm.call")
def test_answer_cross_document_triggers_regeneration_when_review_flags_issue(mock_llm_call):
    mock_llm_call.side_effect = [
        "flawed combined answer that misses a discrepancy",
        "VERDICT: ISSUE\nThe answer does not reconcile the differing confidentiality periods.",
        "a corrected answer that explicitly reconciles the five-year and two-year periods",
    ]

    result = answer_cross_document(
        "How long must confidentiality be kept across these agreements?",
        [NDA_DOC, SERVICE_DOC],
        threshold=0.05,
    )

    assert result.n_llm_calls == 3
    assert result.critique_issue_found is True
    assert result.answer == "a corrected answer that explicitly reconciles the five-year and two-year periods"


@patch("src.llm.call")
def test_answer_cross_document_non_english_document_routes_to_strong(mock_llm_call):
    """Even though cross-document already forces strong, this confirms
    the combined-language signal is computed correctly and doesn't
    break anything when a non-English document is involved."""
    french_doc = make_document(
        "fr_doc",
        {"1. Confidentialite": "Les informations confidentielles doivent etre protegees."},
        language="non_english",
    )
    mock_llm_call.side_effect = ["combined answer", "VERDICT: OK"]

    result = answer_cross_document("compare confidentiality terms", [NDA_DOC, french_doc], threshold=0.05)
    assert result.tier == "strong"


def test_answer_cross_document_requires_at_least_one_document():
    with pytest.raises(ValueError):
        answer_cross_document("any question", [])


# --- load_documents() ---

def test_load_documents_reuses_pipeline_load_document_index(sample_pdf_path):
    loaded = load_documents([sample_pdf_path], encode_fn=fake_encode)
    assert len(loaded) == 1
    assert loaded[0].doc_id  # a real, non-empty deterministic doc_id
    assert loaded[0].full_text  # real extracted text
    assert loaded[0].language == "english"


# --- Routing/logging ---

@patch("src.llm.call")
def test_cross_document_routing_decision_is_logged(mock_llm_call, isolate_log_file):
    mock_llm_call.side_effect = ["combined answer", "VERDICT: OK"]

    answer_cross_document("compare confidentiality terms", [NDA_DOC, SERVICE_DOC], threshold=0.05)

    lines = isolate_log_file.read_text().strip().splitlines()
    routing_entries = [json.loads(line) for line in lines if json.loads(line).get("event") == "routing"]
    assert len(routing_entries) == 1
    assert routing_entries[0]["complexity"] == "cross_document"
    assert routing_entries[0]["tier"] == "strong"
    assert "," in routing_entries[0]["doc_id"]  # combined doc_id for multi-doc questions
