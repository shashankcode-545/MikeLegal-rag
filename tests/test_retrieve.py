"""
Retrieval-vs-full-read decision tests (src/retrieve.py).

Uses the deterministic fake encoder from conftest.py.
"""
import json

from src.index import EmbeddedIndex
from src.models import Chunk
from src.retrieve import decide
from tests.conftest import fake_encode


def make_test_chunks():
    return [
        Chunk(
            doc_id="doc1",
            chunk_id="doc1::c000",
            section_label="1. Termination",
            text="Either party may terminate this agreement with thirty days written notice.",
            page_number=1,
        ),
        Chunk(
            doc_id="doc1",
            chunk_id="doc1::c001",
            section_label="2. Confidentiality",
            text="The recipient shall keep all confidential information secret.",
            page_number=2,
        ),
    ]


def test_successful_retrieval_uses_search_mode():
    chunks = make_test_chunks()
    index = EmbeddedIndex(chunks, encode_fn=fake_encode)
    full_text = " ".join(c.text for c in chunks)

    result = decide(
        question="terminate this agreement notice",
        doc_id="doc1",
        index=index,
        full_document_text=full_text,
        top_k=1,
        threshold=0.1,  # low enough that our fake encoder's overlap clears it
    )

    assert result.mode == "search"
    assert result.n_passages >= 1
    assert result.passages[0].section_label == "1. Termination"


def test_forced_full_document_fallback():
    chunks = make_test_chunks()
    index = EmbeddedIndex(chunks, encode_fn=fake_encode)
    full_text = " ".join(c.text for c in chunks)

    result = decide(
        question="terminate this agreement notice",
        doc_id="doc1",
        index=index,
        full_document_text=full_text,
        top_k=1,
        threshold=0.999,  # unreachable -> forces fallback
    )

    assert result.mode == "full_document"
    assert result.n_passages == 0
    assert result.context == full_text


def test_top_k_and_threshold_are_configurable():
    chunks = make_test_chunks()
    index = EmbeddedIndex(chunks, encode_fn=fake_encode)
    full_text = " ".join(c.text for c in chunks)

    result_k1 = decide(
        "confidentiality secret", "doc1", index, full_text, top_k=1, threshold=0.1
    )
    result_k2 = decide(
        "confidentiality secret", "doc1", index, full_text, top_k=2, threshold=0.1
    )

    assert result_k1.top_k == 1
    assert result_k2.top_k == 2
    assert len(result_k1.passages) <= 1
    assert len(result_k2.passages) <= 2


def test_decision_is_logged_with_required_fields(tmp_path):
    chunks = make_test_chunks()
    index = EmbeddedIndex(chunks, encode_fn=fake_encode)
    full_text = " ".join(c.text for c in chunks)
    log_path = tmp_path / "custom_log.jsonl"

    decide(
        "terminate this agreement notice",
        "doc1",
        index,
        full_text,
        top_k=1,
        threshold=0.1,
        log_path=log_path,
    )

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    required_fields = {
        "doc_id",
        "question",
        "mode",
        "top_score",
        "n_passages",
        "top_k",
        "threshold",
        "timestamp",
    }
    assert required_fields.issubset(entry.keys())
    assert entry["doc_id"] == "doc1"
    assert entry["mode"] == "search"


def test_default_log_path_is_used_when_not_specified(isolate_log_file):
    """isolate_log_file (autouse in conftest) monkeypatches the default
    log path, so this confirms decide() picks up that default correctly
    when no log_path is passed in."""
    chunks = make_test_chunks()
    index = EmbeddedIndex(chunks, encode_fn=fake_encode)
    full_text = " ".join(c.text for c in chunks)

    decide("terminate this agreement notice", "doc1", index, full_text, top_k=1, threshold=0.1)

    assert isolate_log_file.exists()
    lines = isolate_log_file.read_text().strip().splitlines()
    assert len(lines) == 1
