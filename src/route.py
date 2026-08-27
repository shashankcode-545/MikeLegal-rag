"""
Smart model routing.

Given (a) the document's language and (b) how complex the question is,
decide which model TIER should answer -- instead of using one fixed
model for every question. This module never calls an LLM to make that
decision: language detection and complexity classification are both
cheap, rule-based checks, adding zero extra API calls on top of the one
generation call per question.

The routing rule below is a priority-ordered list, not a lookup table
of every (language, complexity) combination. That's a deliberate
choice: the non-English case should apply regardless of complexity (a
mistranslated clause is a correctness problem), so checking language
FIRST and falling through to complexity only for English documents
mirrors the intent more directly than a flat dictionary would.
"""
import json
import re
import time
from pathlib import Path
from typing import Optional

from langdetect import LangDetectException, detect

from src.models import RouteDecision

DEFAULT_LOG_PATH = Path("logs/run_log.jsonl")

# Model tiers. Plain strings on purpose -- llm.py maps these to real
# OpenRouter model ids without needing to import anything from here.
TIER_LIGHT = "light"
TIER_STANDARD = "standard"
TIER_STRONG = "strong"

# Question complexity buckets.
COMPLEXITY_CASUAL = "casual"
COMPLEXITY_SIMPLE_FACTUAL = "simple_factual"
COMPLEXITY_FULL_DOCUMENT_REASONING = "full_document_reasoning"
COMPLEXITY_CONFLICT_OR_OVERRIDE = "conflict_or_override"
# Cross-document comparison routes to the strongest tier, same as
# resolving conflicting/overriding clauses. This constant is only ever
# set directly by cross_doc.py (a cross-document question is known to
# be cross-document by construction, not by classifying question
# text), never by classify_complexity() below.
COMPLEXITY_CROSS_DOCUMENT = "cross_document"

# Language buckets for a document. "none" means no document is loaded
# at all (the casual-conversation case).
LANGUAGE_ENGLISH = "english"
LANGUAGE_NON_ENGLISH = "non_english"
LANGUAGE_MIXED = "mixed"
LANGUAGE_NONE = "none"


# Language detection: run ONCE per document, not per question.

def detect_language(document_text: str) -> str:
    """Classify a document as 'english', 'non_english', or 'mixed'.

    Call this once when a document is first loaded (see
    pipeline.load_document_index) and reuse the result for every
    question asked about that document -- re-running it per question
    would be wasted work for a fact that doesn't change.

    Samples a handful of paragraphs rather than running detection on
    the whole document, since a few paragraphs are enough to tell
    "all English" from "some non-English" and this keeps it fast even
    on long contracts.
    """
    paragraphs = [p.strip() for p in document_text.split("\n\n") if p.strip()]
    sample = paragraphs[:10]

    detected = set()
    for paragraph in sample:
        if len(paragraph) < 20:
            continue  # too short for langdetect to be reliable
        try:
            detected.add(detect(paragraph))
        except LangDetectException:
            continue

    if not detected:
        return LANGUAGE_ENGLISH  # nothing usable detected; safe default
    if detected == {"en"}:
        return LANGUAGE_ENGLISH
    if "en" in detected:
        return LANGUAGE_MIXED
    return LANGUAGE_NON_ENGLISH


# Question complexity classification (run per question).

_CASUAL_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|how are you|"
    r"good morning|good evening|bye|goodbye)\b",
    re.IGNORECASE,
)

_CONFLICT_RE = re.compile(
    r"\b(conflict|override|overrides|prevail|prevails|supersede|supersedes|"
    r"notwithstanding|takes precedence|inconsistent)\b",
    re.IGNORECASE,
)

_SUMMARY_RE = re.compile(
    r"\b(summarize|summarise|overview|entire agreement|whole document|"
    r"walk me through|in general)\b",
    re.IGNORECASE,
)


def classify_complexity(
    question: str, has_document: bool, retrieval_mode: Optional[str]
) -> str:
    """Classify a question into one of four buckets:

      casual                      -- no document grounding needed at all
      conflict_or_override        -- needs to reconcile conflicting/
                                      overriding clauses
      full_document_reasoning     -- broad reasoning over the whole document
      simple_factual              -- a narrow, single-fact lookup

    `retrieval_mode` is the retrieval-vs-full-read decision ("search"
    or "full_document") for this same question. Its fallback to
    reading the whole document is itself a signal that the question
    needed more than a simple lookup.
    """
    if not has_document:
        return COMPLEXITY_CASUAL
    if _CASUAL_RE.match(question.strip()):
        return COMPLEXITY_CASUAL
    if _CONFLICT_RE.search(question):
        return COMPLEXITY_CONFLICT_OR_OVERRIDE
    if _SUMMARY_RE.search(question):
        return COMPLEXITY_FULL_DOCUMENT_REASONING
    if retrieval_mode == "full_document":
        return COMPLEXITY_FULL_DOCUMENT_REASONING
    return COMPLEXITY_SIMPLE_FACTUAL


# Routing decision (language + complexity -> tier).

DEFAULT_TIER = TIER_STANDARD
DEFAULT_REASON = (
    "Language or complexity could not be confidently classified; "
    "using a safe standard-tier default."
)


def route(
    language: str,
    complexity: str,
    doc_id: str = "",
    question: str = "",
    log_path: Optional[Path] = None,
) -> "RouteDecision":
    """Decide which model tier should answer, and log that decision.

    Checked in this order:
      1. Non-English or mixed-language document -> always the strongest
         tier, regardless of how simple the question looks.
      2. No document at all (casual chat) -> lightest tier.
      3. Otherwise, the tier follows the question's complexity.
      4. Anything unrecognized falls through to a safe standard-tier
         default instead of guessing or raising an error.
    """
    if language in (LANGUAGE_NON_ENGLISH, LANGUAGE_MIXED):
        tier = TIER_STRONG
        reason = (
            "Document is in, or mixes in, a non-English language; a "
            "mistranslated clause reference is a correctness problem."
        )
    elif complexity == COMPLEXITY_CASUAL:
        tier = TIER_LIGHT
        reason = "Casual conversation needs no document grounding."
    elif complexity == COMPLEXITY_SIMPLE_FACTUAL:
        tier = TIER_LIGHT
        reason = "Simple, self-contained factual question; no deep reasoning needed."
    elif complexity == COMPLEXITY_FULL_DOCUMENT_REASONING:
        tier = TIER_STANDARD
        reason = "Full-document reasoning balances quality and cost at the standard tier."
    elif complexity == COMPLEXITY_CONFLICT_OR_OVERRIDE:
        tier = TIER_STRONG
        reason = "Resolving conflicting or overriding clauses needs the most capable reasoning tier."
    elif complexity == COMPLEXITY_CROSS_DOCUMENT:
        tier = TIER_STRONG
        reason = "Cross-document comparison needs the most capable reasoning tier."
    else:
        tier, reason = DEFAULT_TIER, DEFAULT_REASON

    decision = RouteDecision(
        doc_id=doc_id,
        question=question,
        language=language,
        complexity=complexity,
        tier=tier,
        reason=reason,
    )
    _log_routing_decision(decision, log_path if log_path is not None else DEFAULT_LOG_PATH)
    return decision


def _log_routing_decision(decision: RouteDecision, log_path: Path) -> None:
    """Append one JSON line per routing decision, separate from
    retrieve.py's log entry. The two lines for a given question share
    doc_id/question and can be correlated by a reader."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "event": "routing",
        "doc_id": decision.doc_id,
        "question": decision.question,
        "language": decision.language,
        "complexity": decision.complexity,
        "tier": decision.tier,
        "reason": decision.reason,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
