# MikeLegal RAG Assistant

A retrieval-augmented legal document assistant, built level by level.
This README is updated as each level is completed; it currently covers
**Level 1 only**.

## Level 1 — Single-Document Question Answering

Given one contract PDF, this pipeline:
1. Extracts text page-by-page (`src/extract.py`)
2. Splits it into clause-level passages (`src/chunk.py`)
3. Embeds each passage and builds an in-memory search index (`src/index.py`)
4. Decides, per question, whether to answer from retrieved passages or
   the full document (`src/retrieve.py`) — logging that decision
5. Sends the resulting context + question to an LLM via OpenRouter for a
   grounded answer (`src/llm.py`)

All five steps are wired together in `src/pipeline.py`.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# then edit .env and add your real OPENROUTER_API_KEY
```

The first time `EmbeddedIndex` runs with the real (default) encoder, it
downloads `all-MiniLM-L6-v2` from Hugging Face (a few hundred MB, one
time only, then cached locally). This requires normal internet access
to `huggingface.co`.

### Running it

Ask a question against one of the sample contracts in `data/contracts/`:

```bash
python3 -c "
from src.pipeline import answer_single_document

result = answer_single_document(
    question='What is the notice period for termination?',
    pdf_path='data/contracts/NON disclosure agreement Edited.pdf',
)
print('mode:', result.mode)
print('confidence:', result.confidence)
print('answer:', result.answer)
"
```

To ask a second question about the same document without re-embedding
it, load the index once and reuse it:

```bash
python3 -c "
from src.pipeline import load_document_index, answer_single_document

doc_id, chunks, index, full_text = load_document_index(
    'data/contracts/NON disclosure agreement Edited.pdf'
)

for question in ['Who are the parties?', 'What happens on breach?']:
    result = answer_single_document(
        question, doc_id=doc_id, index=index, full_text=full_text
    )
    print(question, '->', result.mode, '->', result.answer)
"
```

### Configuring retrieval sensitivity

`top_k` (how many passages to pull) and `threshold` (how confident a
match must be before it's trusted) are plain parameters on
`answer_single_document(...)` / `retrieve.decide(...)` — pass different
values per call, no config file or restart needed:

```python
answer_single_document(question, pdf_path=path, top_k=5, threshold=0.5)
```

If no chunk clears `threshold`, the pipeline automatically falls back
to reading the entire document instead of returning a weak match.

### Where decisions are logged

Every retrieval decision (search vs. full-document, confidence score,
number of passages, the `top_k`/`threshold` used) is appended as one
JSON line to `logs/run_log.jsonl`. This log is the input Level 2's
routing will read from.

### Running the tests

```bash
pytest tests/ -v
```

Tests never make real OpenRouter calls (the LLM is mocked) and never
require the real embedding model or network access — they use a small
deterministic fake encoder (see `tests/conftest.py`) so retrieval-logic
tests (thresholds, top_k, logging) run fast and offline. Real chunking
and extraction tests run against an actual sample contract in
`data/contracts/`.

```
13 tests: extraction/chunking metadata, deterministic chunk IDs,
search vs. fallback retrieval modes, configurable top_k/threshold,
JSONL logging fields, and the end-to-end pipeline with a mocked LLM.
```

## Level 2 — Smart Model Routing

Adds routing on top of Level 1: instead of always using one fixed
model, each question is answered by one of three model **tiers**
(`light` / `standard` / `strong`), chosen by two cheap, rule-based
signals -- no extra LLM call is spent deciding this.

- **Document language** (`src/route.py: detect_language`) is detected
  **once per document** (inside `load_document_index`) and reused for
  every question about that document.
- **Question complexity** (`src/route.py: classify_complexity`) is
  classified per question as `casual`, `simple_factual`,
  `full_document_reasoning`, or `conflict_or_override` -- reusing
  Level 1's own retrieval-vs-full-read `mode` as one of the signals.
- **`route()`** combines both into a tier + a short reason, following
  the same priority the assignment specifies: a non-English or
  mixed-language document always routes to the strongest tier
  regardless of complexity; otherwise the tier follows complexity.
  An unrecognized case safely falls back to the standard tier.
- **`llm.resolve_model_for_tier(tier)`** maps a tier name to a real
  OpenRouter model id, read from `MODEL_TIER_LIGHT` /
  `MODEL_TIER_STANDARD` / `MODEL_TIER_STRONG` in `.env` (same model
  family for all three, per the assignment).

Every routing decision is logged to `logs/run_log.jsonl` as its own
JSON line (`"event": "routing"`) alongside Level 1's retrieval log
line, so both decisions for a given question can be seen together.

### Running Level 2

Ask a question against a document, letting routing pick the tier:

```bash
python3 -c "
from src.pipeline import answer_single_document

result = answer_single_document(
    question='What is the notice period for termination?',
    pdf_path='data/contracts/NON disclosure agreement Edited.pdf',
)
print('tier:', result.tier, '-', result.route_reason)
print('answer:', result.answer)
"
```

Casual, document-independent questions use a separate function that
skips retrieval entirely:

```bash
python3 -c "
from src.pipeline import answer_casual

result = answer_casual('hey, how are you?')
print('tier:', result.tier, '-', result.route_reason)
print('answer:', result.answer)
"
```

Example of a routed, non-English question (documented per the
assignment's Definition of Done) -- a French-language document routes
to the strong tier even for a simple factual question, because a
mistranslated clause reference is a correctness problem:

```python
from src.pipeline import answer_single_document
from src.index import EmbeddedIndex
from src.models import Chunk

chunks = [Chunk(doc_id="fr_doc", chunk_id="fr_doc::c000",
                 section_label="1. Resiliation",
                 text="Chaque partie peut resilier cet accord avec un preavis de trente jours.",
                 page_number=1)]
index = EmbeddedIndex(chunks)  # real embedding model
result = answer_single_document(
    "What is the notice period?",
    doc_id="fr_doc", index=index, full_text=chunks[0].text,
    language="non_english",
)
print(result.tier)  # -> "strong"
```

### Running the tests

```bash
pytest tests/ -v
```

Level 2 adds `tests/test_route.py` (language detection, complexity
classification, all five routing situations from the assignment plus
the safe-default fallback, JSONL logging format, tier→model env var
resolution) and extends `tests/test_pipeline.py` with routing wired
into the full pipeline, language-detection caching, and confirmation
that the routed model id actually reaches `llm.call()`. All Level 1
tests continue to pass unchanged -- Level 2 added new behavior without
modifying `extract.py`, `chunk.py`, `index.py`, or `retrieve.py`.

```
42 tests total (13 from Level 1 + 29 new/extended for Level 2).
```

## Level 3 — Fusion and Critique for Complex Reasoning

Adds one independent review pass for reasoning-heavy questions --
specifically, the questions Level 2 already classifies as
`conflict_or_override` (resolving conflicting/overriding clauses).
Every other question type (casual, simple factual, full-document
reasoning) is completely untouched and still costs exactly one LLM call.

- **`src/critique.py: review()`** — one call, in a distinct reviewer
  system-prompt role, checking the draft against the source text for
  an unaddressed carve-out/override, an unsupported claim, or an
  unreconciled contradiction between clauses. Returns a clear verdict
  (`VERDICT: OK` or `VERDICT: ISSUE` + explanation).
- **`src/critique.py: regenerate()`** — only called if the review flags
  a real issue. One regeneration attempt with the reviewer's feedback
  folded in. If the model still can't resolve it, it prefixes
  `"UNRESOLVED:"` in its own response, so the caller can tell without a
  fourth call -- the original draft is sent back in that case, with the
  discrepancy recorded rather than silently dropped.
- **`src/critique.py: answer_with_critique()`** — orchestrates the
  above and reports exactly how many calls a question required:
  - not reasoning-heavy → **1 call** (draft only)
  - reasoning-heavy, review says OK → **2 calls**
  - reasoning-heavy, issue found, regenerated → **3 calls**

Reviewer and regenerator reuse the same strong-tier model the draft
already used (conflict/override questions are already routed to the
strong tier by Level 2) in a genuinely separate role -- a distinct
system prompt that never sees the draft's own reasoning, only its
output. Every critique decision is logged to `logs/run_log.jsonl` as
its own line (`"event": "critique"`), including the 1-call case, so
the added cost per question is directly measurable.

### Running Level 3

```bash
python3 -c "
from src.pipeline import answer_single_document

result = answer_single_document(
    question='Which clause takes precedence if two sections conflict?',
    pdf_path='data/contracts/NON disclosure agreement Edited.pdf',
)
print('n_llm_calls:', result.n_llm_calls)
print('issue_found:', result.critique_issue_found)
print('unresolved:', result.critique_unresolved)
print('answer:', result.answer)
"
```

A simple factual question against the same document will show
`n_llm_calls: 1` -- critique never runs for it.

### Running the tests

```bash
pytest tests/ -v
```

Level 3 adds `tests/test_critique.py`, including the Definition of
Done's planted-conflict scenario: a small contract-like snippet with
two clauses that genuinely and permanently contradict each other,
where the review step must catch the conflict and trigger a revised
answer that reconciles it (and a companion test for the case where
regeneration still can't resolve it). `tests/test_pipeline.py` is
extended to confirm the same behavior end-to-end against a real sample
contract. No real OpenRouter calls are made in any test.

```
51 tests total (13 from Level 1 + 29 from Level 2 + 9 new for Level 3).
```

## Level 4 — Cross-Document Citation Grounding

Extends the assistant to answer questions that span **multiple**
documents (comparing terms, reconciling clauses across contracts),
with every citation in the final answer traceable back to a specific
retrieved passage.

- **`src/cross_doc.py: load_documents(pdf_paths)`** — loads/embeds each
  document once (reusing Level 1's `load_document_index` unchanged), so
  a set of documents can be loaded once and reused across many
  cross-document questions.
- **`src/cross_doc.py: extract_from_documents(question, documents)`** —
  runs Level 1's own `retrieve.decide()` once per document (no LLM
  calls -- pure embedding search). A confident match cites the exact
  retrieved chunk's own `chunk_id` (e.g. `some_doc::c003`); a
  full-document fallback cites `"{doc_id}::full_document"` instead.
- **`src/cross_doc.py: answer_cross_document(question, documents)`** —
  builds one fusion prompt from all the tagged excerpts, makes exactly
  **one** LLM call to produce a combined answer, routes it through
  Level 2 (cross-document comparison always maps to the strong tier,
  per the assignment's own routing table) and then through Level 3's
  critique flow unchanged (cross-document is explicitly one of the two
  cases Level 3 was built for) -- adding 0-2 more calls for review and
  a possible single regeneration. **Total: 1-3 LLM calls, regardless of
  how many documents are being compared.**
- **`src/cross_doc.py: verify_citations(answer, extractions)`** — checks
  every `[bracketed]` citation in the final answer against what was
  actually retrieved. An unverifiable citation (a fabricated document
  or passage) is recorded as a warning on the result -- **the request
  still succeeds**, nothing is silently dropped or failed.

### Running Level 4

```bash
python3 -c "
from src.cross_doc import load_documents, answer_cross_document

docs = load_documents([
    'data/contracts/NON disclosure agreement Edited.pdf',
    'data/contracts/Service agreement new clm Copy.pdf',
])
result = answer_cross_document(
    'Compare the confidentiality terms across these two agreements.',
    docs,
)
print('tier:', result.tier)
print('n_llm_calls:', result.n_llm_calls)
print('citation_warnings:', result.citation_warnings)
print('answer:', result.answer)
"
```

Load the documents once, then reuse `docs` for as many follow-up
cross-document questions as needed -- no re-extraction or re-embedding.

### Running the tests

```bash
pytest tests/ -v
```

Level 4 adds `tests/test_cross_doc.py`, including the Definition of
Done's two required cases: a fully-traceable combined answer (zero
citation warnings) and a deliberately fabricated citation that gets
flagged as a warning without failing the request. Tests also confirm
LLM call count stays at 1-3 regardless of whether 2 or 3+ documents are
being compared, using real sample contracts where practical. No real
OpenRouter calls are made in any test.

```
68 tests total (13 Level 1 + 29 Level 2 + 9 Level 3 + 17 new for Level 4).
```
