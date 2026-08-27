"""
Smart model routing.

Given (a) the document's language and (b) the question's complexity,
decide which model tier should answer. Language detection and complexity
classification are rule-based, adding zero LLM calls.

The routing rule is priority-ordered: language is checked first,
complexity only for English documents.
"""
import json
import re
import time
from pathlib import Path
from typing import Optional

from langdetect import LangDetectException, detect

from src.models import RouteDecision

DEFAULT_LOG_PATH = Path("logs/run_log.jsonl")

TIER_LIGHT = "light"
TIER_STANDARD = "standard"
TIER_STRONG = "strong"

COMPLEXITY_CASUAL = "casual"
COMPLEXITY_SIMPLE_FACTUAL = "simple_factual"
COMPLEXITY_FULL_DOCUMENT_REASONING = "full_document_reasoning"
COMPLEXITY_CONFLICT_OR_OVERRIDE = "conflict_or_override"
# Set directly by cross_doc.py, never by classify_complexity().
COMPLEXITY_CROSS_DOCUMENT = "cross_document"

LANGUAGE_ENGLISH = "english"
LANGUAGE_NON_ENGLISH = "non_english"
LANGUAGE_MIXED = "mixed"
LANGUAGE_NONE = "none"



def detect_language(document_text: str) -> str:
    """Classify a document as 'english', 'non_english', or 'mixed'.

    Samples a handful of paragraphs rather than the whole document."""
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
    """Classify a question into one of four complexity buckets.

    `retrieval_mode` ("search" or "full_document") is used as an
    additional signal: a full-document fallback implies the question
    needed more than a simple lookup."""
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

    Checked in this order (matches the assignment's routing table):
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
    """Append one JSON line per routing decision."""
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
