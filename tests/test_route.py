"""
Routing tests (src/route.py): detect_language(), classify_complexity(),
route(), and the tier->model mapping in llm.py.
"""
import json

from src import llm
from src.route import (
    COMPLEXITY_CASUAL,
    COMPLEXITY_CONFLICT_OR_OVERRIDE,
    COMPLEXITY_FULL_DOCUMENT_REASONING,
    COMPLEXITY_SIMPLE_FACTUAL,
    DEFAULT_REASON,
    DEFAULT_TIER,
    TIER_LIGHT,
    TIER_STANDARD,
    TIER_STRONG,
    classify_complexity,
    detect_language,
    route,
)


# --- Language detection ---

ENGLISH_TEXT = (
    "This Non-Disclosure Agreement is entered into by and between the "
    "parties below.\n\n"
    "Each party agrees to keep confidential information secret and not "
    "to disclose it to any third party without prior written consent.\n\n"
    "This agreement shall remain in effect for a period of two years "
    "from the date of signing, unless terminated earlier by either party."
)

# Genuinely non-English (French) contract-like text.
FRENCH_TEXT = (
    "Le present accord de confidentialite est conclu entre les parties "
    "mentionnees ci-dessous.\n\n"
    "Chaque partie s'engage a garder les informations confidentielles "
    "secretes et a ne pas les divulguer a un tiers sans consentement "
    "prealable ecrit.\n\n"
    "Cet accord restera en vigueur pendant une periode de deux ans a "
    "compter de la date de signature, sauf resiliation anticipee."
)

# A document that mixes English and French paragraphs.
MIXED_TEXT = (
    "This Non-Disclosure Agreement is entered into by and between the "
    "parties below, and each party agrees to keep confidential "
    "information secret from third parties.\n\n"
    "Chaque partie s'engage a garder les informations confidentielles "
    "secretes et a ne pas les divulguer a un tiers sans consentement "
    "prealable ecrit du proprietaire de ces informations.\n\n"
    "This agreement shall remain in effect for a period of two years "
    "from the date of signing, unless terminated earlier by either party."
)


def test_detect_language_english_document():
    assert detect_language(ENGLISH_TEXT) == "english"


def test_detect_language_non_english_document():
    assert detect_language(FRENCH_TEXT) == "non_english"


def test_detect_language_mixed_document():
    assert detect_language(MIXED_TEXT) == "mixed"


def test_detect_language_empty_or_unusable_text_defaults_to_english():
    """No usable paragraphs (too short/empty) -> safe default rather
    than crashing or guessing."""
    assert detect_language("") == "english"
    assert detect_language("hi\n\nok") == "english"  # all fragments too short


# --- Complexity classification ---

def test_classify_casual_when_no_document():
    complexity = classify_complexity("hello there", has_document=False, retrieval_mode=None)
    assert complexity == COMPLEXITY_CASUAL


def test_classify_casual_greeting_even_with_document():
    complexity = classify_complexity("hi!", has_document=True, retrieval_mode="search")
    assert complexity == COMPLEXITY_CASUAL


def test_classify_simple_factual_on_confident_search_hit():
    complexity = classify_complexity(
        "What is the notice period?", has_document=True, retrieval_mode="search"
    )
    assert complexity == COMPLEXITY_SIMPLE_FACTUAL


def test_classify_full_document_reasoning_on_fallback_mode():
    """Level 1's fallback to full-document mode is itself reused here
    as a complexity signal, per the assignment."""
    complexity = classify_complexity(
        "What does this contract cover?", has_document=True, retrieval_mode="full_document"
    )
    assert complexity == COMPLEXITY_FULL_DOCUMENT_REASONING


def test_classify_full_document_reasoning_on_summary_keyword():
    complexity = classify_complexity(
        "Can you summarize the entire agreement?", has_document=True, retrieval_mode="search"
    )
    assert complexity == COMPLEXITY_FULL_DOCUMENT_REASONING


def test_classify_conflict_or_override_on_keyword():
    complexity = classify_complexity(
        "Which clause takes precedence if section 3 and section 7 conflict?",
        has_document=True,
        retrieval_mode="search",
    )
    assert complexity == COMPLEXITY_CONFLICT_OR_OVERRIDE


# --- Routing decision ---

def test_routing_non_english_document(tmp_path):
    decision = route(
        language="non_english",
        complexity=COMPLEXITY_SIMPLE_FACTUAL,  # even a "simple" question
        log_path=tmp_path / "log.jsonl",
    )
    assert decision.tier == TIER_STRONG


def test_routing_mixed_language_document(tmp_path):
    decision = route(
        language="mixed",
        complexity=COMPLEXITY_SIMPLE_FACTUAL,
        log_path=tmp_path / "log.jsonl",
    )
    assert decision.tier == TIER_STRONG


def test_routing_simple_factual_english(tmp_path):
    decision = route(
        language="english",
        complexity=COMPLEXITY_SIMPLE_FACTUAL,
        log_path=tmp_path / "log.jsonl",
    )
    assert decision.tier == TIER_LIGHT


def test_routing_full_document_reasoning_english(tmp_path):
    decision = route(
        language="english",
        complexity=COMPLEXITY_FULL_DOCUMENT_REASONING,
        log_path=tmp_path / "log.jsonl",
    )
    assert decision.tier == TIER_STANDARD


def test_routing_conflict_or_override(tmp_path):
    decision = route(
        language="english",
        complexity=COMPLEXITY_CONFLICT_OR_OVERRIDE,
        log_path=tmp_path / "log.jsonl",
    )
    assert decision.tier == TIER_STRONG


def test_routing_casual_no_document(tmp_path):
    decision = route(
        language="none",
        complexity=COMPLEXITY_CASUAL,
        log_path=tmp_path / "log.jsonl",
    )
    assert decision.tier == TIER_LIGHT


def test_routing_unrecognized_input_uses_safe_default(tmp_path):
    """A language/complexity pair that doesn't match any rule (e.g.
    classification produced something unexpected) must not crash --
    it should fall through to the documented safe default."""
    decision = route(
        language="klingon",
        complexity="some_unrecognized_value",
        log_path=tmp_path / "log.jsonl",
    )
    assert decision.tier == DEFAULT_TIER
    assert decision.reason == DEFAULT_REASON


def test_routing_decision_is_logged(tmp_path):
    log_path = tmp_path / "log.jsonl"
    route(
        language="english",
        complexity=COMPLEXITY_SIMPLE_FACTUAL,
        doc_id="doc1",
        question="What is the notice period?",
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
        "language",
        "complexity",
        "tier",
        "reason",
    }
    assert required_fields.issubset(entry.keys())
    assert entry["event"] == "routing"
    assert entry["tier"] == TIER_LIGHT


# --- Tier -> model id mapping ---

def test_resolve_model_for_tier_uses_env_var_when_set(monkeypatch):
    monkeypatch.setenv("MODEL_TIER_LIGHT", "custom/light-model")
    assert llm.resolve_model_for_tier("light") == "custom/light-model"


def test_resolve_model_for_tier_falls_back_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("MODEL_TIER_STANDARD", raising=False)
    resolved = llm.resolve_model_for_tier("standard")
    assert resolved == llm._TIER_DEFAULTS["standard"]


def test_resolve_model_for_tier_unrecognized_tier_falls_back_to_standard(monkeypatch):
    monkeypatch.delenv("MODEL_TIER_STANDARD", raising=False)
    resolved = llm.resolve_model_for_tier("not_a_real_tier")
    assert resolved == llm._TIER_DEFAULTS["standard"]
