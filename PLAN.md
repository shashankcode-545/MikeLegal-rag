# PLAN.md

This file is written before code for each level and updated (not replaced)
as each new level begins. Each section below represents my understanding
and intended approach at the time that level was started.

---

## Level 1 — Single-Document Question Answering

### Understanding of the goal

Given one uploaded contract, build a pipeline that:
1. Splits the document into passages (chunks).
2. Converts each passage into a searchable (embedded) representation and
   stores them.
3. Given a question, decides whether to **search** for the most relevant
   passage(s) or **fall back to reading the whole document** — this
   decision must be configurable per request (not hardcoded), and every
   decision must be logged with: how many passages were found, how
   confident the match was, and whether it fell back to a full read.
4. Generates an answer grounded in whatever text was retrieved.

The retrieval-vs-full-read decision and its log are explicitly called out
as "the quality signal every later level depends on" — Level 2's routing
table consumes this signal directly, so it needs to be a clean, reusable
function, not something buried inline in a script.

### Architecture

```
PDF file
   │  extract.py (pypdf)
   ▼
raw text
   │  chunk.py (split on numbered clause headings, e.g. "1.", "1.1",
   │            "ARTICLE II"; fixed word-count fallback for unstructured
   │            sections)
   ▼
list[Chunk(doc_id, chunk_id, section_label, text)]
   │  index.py (embed each chunk once with a local sentence-transformer
   │            model; store as an in-memory numpy matrix per document)
   ▼
embedded chunk store
   │
   │  question comes in ──▶ retrieve.py:
   │     1. embed the question
   │     2. cosine-similarity search against the doc's chunk matrix
   │     3. if best match score ≥ threshold AND at least one hit found:
   │          mode = "search", return top_k passages
   │        else:
   │          mode = "full_document", return entire document text
   │     4. log {doc_id, question, mode, top_score, n_passages, top_k,
   │             threshold} to logs/run_log.jsonl
   ▼
retrieved context (passages OR full document)
   │  llm.py (single OpenRouter call: question + context → answer)
   ▼
grounded answer
```

### Key design decisions (and why)

- **No vector database.** With ~16 sample documents and an expected
  handful of chunks each, the entire embedding store fits trivially in
  memory as a numpy array. A vector DB (FAISS/Chroma/Pinecone) would add
  a dependency and setup complexity with zero practical benefit at this
  scale. The 300-document scaling question in REFLECTIONS.md is where a
  real index (e.g. FAISS) gets justified instead.
- **Embeddings run locally** (sentence-transformers, all-MiniLM-L6-v2)
  rather than via OpenRouter. This costs nothing and keeps the entire
  $3 OpenRouter budget available for the parts of the assignment that
  actually require model judgment (answering, routing, critique,
  fusion). Only generation calls hit the API.
- **Chunking is clause-aware.** Legal documents are already segmented by
  their own numbering (1., 1.1, Article II, etc.). Splitting along those
  boundaries produces chunks that are more meaningful to cite later
  (Level 4 needs passage-level citations) than an arbitrary fixed-size
  token window. A word-count fallback handles any unstructured text that
  doesn't match a heading pattern.
- **Retrieval sensitivity is just function parameters** (`top_k`,
  `threshold` on `retrieve.decide(...)`), not a config file or UI toggle.
  This satisfies "configurable per request" with the least possible
  machinery, and each call's exact parameters are captured in the log
  entry for that request — so a reviewer can see the default was used,
  or that it was overridden, directly from the logs.
- **The full-document fallback is a deliberate, logged branch**, not
  silent behavior — if nothing in the index is a confident match, the
  system explicitly says so (`mode="full_document"`, with a reason) and
  reads the whole document instead of returning a weak/low-confidence
  passage.

### Files / functions for this level

- `src/extract.py` — `extract_text(pdf_path) -> str`
- `src/chunk.py` — `chunk_document(text, doc_id) -> list[Chunk]`
- `src/models.py` — `Chunk`, `RetrievalResult` dataclasses
- `src/index.py` — `embed_chunks(chunks) -> EmbeddedIndex`,
  `EmbeddedIndex.search(query, top_k) -> list[(Chunk, score)]`
- `src/retrieve.py` — `decide(question, doc_id, index, top_k, threshold) -> RetrievalResult`
  (also owns writing the log entry)
- `src/llm.py` — `call(model_id, system_prompt, user_prompt) -> str`
  (thin OpenRouter wrapper; tier→model mapping arrives in Level 2, but the
  raw call function is built now since Level 1 needs to generate an answer)
- `src/pipeline.py` — `answer_single_document(question, doc_id, ...) -> Answer`
  (wires extract → chunk → index → retrieve → llm together)

### Testing approach

`tests/test_level1_retrieval.py`:
- **Search-hit case:** ask a specific, narrow factual question that should
  closely match one clause in a real sample contract (e.g. a notice-period
  or payment-term question). Assert `mode == "search"`, `n_passages >= 1`,
  and the answer text reflects the matched clause.
- **Fallback case:** force a fallback deliberately (e.g. an artificially
  high `threshold` that no chunk can meet, or a deliberately vague/broad
  question) and assert `mode == "full_document"` with a logged reason.
- Both tests assert a log entry was actually written to
  `logs/run_log.jsonl` with the expected fields — this is the "visible
  log entry" required by the Definition of Done.
- LLM calls in tests are mocked (fixed fake responses) so tests don't
  depend on OpenRouter credits or network access; one separate, sparingly-run
  manual smoke test exercises a real OpenRouter call end-to-end.

---

## Level 2 — Smart Model Routing

### Understanding of the goal

Not every question needs the same amount of computation. Level 2 sits
between Level 1's retrieval decision and the LLM call, and decides
*which model tier* answers -- instead of always using the same fixed
model. Two signals drive that decision:

1. **Document language** (English / non-English / mixed) -- detected
   once per document and reused for every question about it.
2. **Question complexity** (casual / simple factual / full-document
   reasoning / conflict-or-override) -- classified per question, reusing
   Level 1's own retrieval-vs-full-read decision as one of the signals.

Both signals are produced by cheap, rule-based logic (a small language-
detection library + regex pattern matching) -- no LLM call is spent on
routing itself. This keeps Level 2's total API cost identical to
Level 1's: exactly one generation call per question.

### Architecture

```
load_document_index(pdf_path)              [Level 1, extended]
  extract -> chunk -> embed
  + detect_language(full_text)  ---- NEW: runs ONCE per document
  -> (doc_id, chunks, index, full_text, language)

answer_single_document(question, doc_id, index, full_text, language)
  retrieve.decide(...)                       [Level 1, unchanged]
      -> RetrievalResult(mode, top_score, passages, ...)
  classify_complexity(question, has_document=True, retrieval_mode)  NEW
      -> "casual" | "simple_factual" | "full_document_reasoning"
         | "conflict_or_override"
  route(language, complexity)                NEW
      -> RouteDecision(tier, reason)  (+ logged)
  llm.resolve_model_for_tier(tier)           NEW
      -> real OpenRouter model id
  llm.call(question, context, model=...)     [Level 1, unchanged call,
                                               now passed a routed model]
  -> Answer(..., tier=, route_reason=)
```

A new `answer_casual(question)` function in `pipeline.py` handles the
one situation that has no document at all (the assignment's "casual
conversation, no document" case): it skips `retrieve.decide()` entirely
since there's nothing to search, and routes straight to the light tier.

### Key design decisions (and why)

- **Language detection runs once per document, not once per question.**
  `load_document_index()` (Level 1's existing one-time-per-document
  loader) now also calls `detect_language()` and returns it alongside
  the index. Callers keep and reuse the returned `language` value for
  every subsequent question, the same way they already reuse the
  embedding index.
- **Routing is a priority-ordered if/elif chain, not a lookup table of
  every (language, complexity) combination.** The assignment's own
  routing table has the non-English row apply "regardless of
  complexity" ("a mistranslated clause reference is a correctness
  problem") -- so language is checked first, and complexity only
  decides the tier for English documents. A flat dict keyed by
  `(language, complexity)` would either have to duplicate the
  non-English row four times or lose that priority relationship; the
  explicit if/elif chain says exactly what the assignment's table says,
  in the same order, and is easy to read top-to-bottom.
- **No LLM call for classification.** Language detection uses
  `langdetect` (a small, deterministic library). Complexity
  classification is a handful of regex checks plus reuse of Level 1's
  own `mode` field (`"full_document"` fallback is itself treated as a
  signal that the question needed more than a lookup). This is what
  keeps Level 2 from adding any API cost on top of Level 1.
- **An unrecognized (language, complexity) pair falls through to a
  logged, safe standard-tier default** rather than raising or guessing
  -- this is the "classification fails" case the Definition of Done
  asks for.
- **Tier names are decoupled from real model ids.** `route.py` only
  ever produces the strings `"light"` / `"standard"` / `"strong"`.
  `llm.resolve_model_for_tier()` is the one place that maps a tier to
  an actual OpenRouter model id (read from `.env`, same model family
  for all three tiers), so `route.py` never needs to know what models
  exist and `llm.py` never needs to know why a tier was picked.
- **Routing decisions are logged separately from Level 1's retrieval
  log**, as their own JSON lines (`"event": "routing"`) in the same
  `logs/run_log.jsonl` file, correlated by `doc_id` + `question`. This
  meant zero changes to `retrieve.py`'s existing logging code.
- **`model=` remains a valid override** on `answer_single_document()` /
  `answer_casual()` for tests or manual use -- if supplied, it bypasses
  the resolved tier's model but routing still runs and still gets
  logged, so the decision is always visible even when overridden.

### Files / functions for this level

- `src/route.py` (new) — `detect_language(text) -> str`,
  `classify_complexity(question, has_document, retrieval_mode) -> str`,
  `route(language, complexity, ...) -> RouteDecision` (also logs)
- `src/models.py` — added `RouteDecision` dataclass; extended `Answer`
  with optional `tier` / `route_reason` fields
- `src/llm.py` — added `resolve_model_for_tier(tier) -> str`,
  `CASUAL_SYSTEM_PROMPT`; `call()` gained an optional `system_prompt`
  override and now handles empty `context` gracefully (unchanged
  default behavior otherwise)
- `src/pipeline.py` — `load_document_index()` now also returns
  `language`; `answer_single_document()` now classifies complexity,
  routes, and resolves the tier's model before calling the LLM;
  `answer_casual()` is new

### Testing approach

`tests/test_route.py` (new) unit-tests `route.py` in isolation:
language detection on real English/French/mixed sample text,
complexity classification for all four buckets, routing output for all
five assignment situations plus the safe-default fallback case, the
JSONL log format, and tier→model-id resolution (env var + fallback).

`tests/test_pipeline.py` (extended) covers routing wired into the full
flow: simple-factual → light, full-document fallback → standard,
conflict → strong, non-English document → strong regardless of
complexity, casual/no-document → light, language detected once and
reused (not re-detected per question), the routed model id actually
reaching `llm.call()`, and the explicit `model=` override still
working. All original Level 1 tests (`test_chunk.py`, `test_retrieve.py`,
and the original `test_pipeline.py` cases) continue to pass unchanged.

---

## Level 3 — Fusion and Critique for Complex Reasoning

### Understanding of the goal

For the hardest questions -- the ones Level 2 already flags as
`conflict_or_override` ("resolving conflicting/overriding clauses") --
a single unchecked draft answer is a risk: a model can claim to have
reconciled two clauses without actually doing so. Level 3 adds one
independent review pass over the draft, and, only if that review finds
a real problem, exactly one regeneration attempt with the reviewer's
feedback folded in. Simple, casual, and full-document-reasoning
questions are never touched by this -- they still cost exactly the one
LLM call they cost in Level 1/2.

### Architecture

```
answer_single_document(...)
  ... retrieve, classify_complexity, route ...   [Level 1/2, unchanged]
  llm.call(question, context, model=resolved_model)   -> draft_answer  [1 call]

  is_reasoning_heavy = (complexity == "conflict_or_override")

  critique.answer_with_critique(question, context, draft_answer,
                                 is_reasoning_heavy, reviewer_model, regenerate_model)
      if not is_reasoning_heavy:
          -> final_answer = draft_answer, n_calls = 1                    [0 extra calls]
      else:
          review(...)                          -> verdict text          [+1 call]
          if verdict says OK:
              -> final_answer = draft_answer, n_calls = 2
          else:
              regenerate(..., feedback)         -> revised text          [+1 call]
              if revised text starts with "UNRESOLVED:":
                  -> final_answer = draft_answer (original), n_calls = 3, unresolved=True
              else:
                  -> final_answer = revised text, n_calls = 3, unresolved=False

  -> Answer(answer=final_answer, ..., n_llm_calls=, critique_issue_found=, critique_unresolved=)
```

### Key design decisions (and why)

- **The critique trigger is Level 2's own complexity flag, not the
  model tier.** `is_reasoning_heavy = (complexity == "conflict_or_override")`.
  This is deliberately narrower than "tier == strong", because a
  non-English document also routes to the strong tier for a completely
  different reason (translation risk) that has nothing to do with
  reconciling clauses -- reviewing those questions wouldn't match what
  the assignment means by "reasoning-heavy."
- **Exactly one regeneration attempt, no re-review.** "Allow only one
  round" is read literally: after regenerating once, the answer is
  shipped without spending a fourth call to check whether the
  regeneration actually worked. Instead, the regeneration prompt itself
  asks the model to prefix `"UNRESOLVED:"` if it still can't fix the
  issue -- so "fixed" vs. "still broken" is read off that single call's
  own text, and the original draft is sent back (with the discrepancy
  recorded) exactly when the assignment says to: "if the second attempt
  still has issues, send the original answer but record that a
  discrepancy was found."
- **Call-count budget is small and fixed**, matching the Definition of
  Done's "a simple question takes one pass, a reasoning-heavy question
  takes more": 1 call (not reasoning-heavy), 2 calls (reasoning-heavy,
  review says OK), or 3 calls (reasoning-heavy, issue found, one
  regeneration). No path ever reaches a 4th call.
- **Reviewer and regenerator reuse the same strong-tier model as the
  draft, in a genuinely distinct role.** The assignment asks for "a
  stronger model in a distinct reviewer role." Questions that reach
  this module were already routed to the strong tier by Level 2 (the
  assignment's own table lists conflict/override as one of the two
  situations that route there), and elsewhere the assignment restricts
  routing to a single model family -- so rather than inventing a
  fictional fourth, even-stronger tier, "distinct role" is implemented
  as a completely separate system prompt (`REVIEWER_SYSTEM_PROMPT`)
  that frames the call as an independent check, never shown the
  draft's own reasoning, only its output.
- **`draft_answer` is passed into `answer_with_critique()`, not
  generated inside it.** `critique.py` never needs to know how the
  draft was produced -- `pipeline.py` already made that one call as
  part of the normal Level 1/2 flow. This keeps `critique.py` reusable
  and independently testable without touching retrieval or routing at
  all.
- **Critique decisions are logged separately**, as their own JSON lines
  (`"event": "critique"`) in the same `logs/run_log.jsonl`, following
  the same per-module logging pattern as `retrieve.py` and `route.py`.
  Every question that goes through `answer_with_critique()` gets a log
  line -- including the 1-call, never-reviewed case -- so the log shows
  exactly how many calls *every* question required, which is what
  "measurable added cost" means here.

### Files / functions for this level

- `src/critique.py` (implemented) — `review(question, draft_answer,
  source_text, model) -> str`, `regenerate(question, context, feedback,
  model) -> str`, `answer_with_critique(question, context, draft_answer,
  is_reasoning_heavy, reviewer_model, regenerate_model, ...) ->
  CritiqueResult` (also logs)
- `src/models.py` — extended `Answer` with `n_llm_calls`,
  `critique_issue_found`, `critique_unresolved` (all optional, default
  `None`, so Level 1/2 code paths are unaffected)
- `src/pipeline.py` — `answer_single_document()` now generates the
  draft, determines `is_reasoning_heavy` from the complexity Level 2
  already classified, and passes the draft through
  `critique.answer_with_critique()` before returning. `answer_casual()`
  is untouched except for reporting `n_llm_calls=1` (it can never be
  reasoning-heavy, since a casual question has no document to find a
  conflict in).

### Testing approach

`tests/test_critique.py` (new) unit-tests `critique.py` in isolation,
with `llm.call` mocked throughout (no real OpenRouter calls, no API
cost):
- a non-reasoning-heavy question never triggers review (0 extra calls)
- a reasoning-heavy question where review says OK (2 calls total)
- **the Definition of Done's planted-conflict test**: a small
  contract-like text with two clauses that genuinely and permanently
  contradict each other (one says confidentiality survives termination
  indefinitely, the other says ALL obligations, explicitly including
  that clause, terminate immediately) -- review is mocked to catch the
  conflict, regeneration is mocked to produce a response that
  acknowledges and reconciles it, and the test asserts the final answer
  changed from the flawed draft to the corrected one
- a companion test where the regeneration attempt starts with
  `"UNRESOLVED:"` -- confirms the original draft is sent back and
  `unresolved=True` is recorded, not a fourth call
- an explicit call-count comparison test (1 vs. 3) for the "simple
  takes one pass, reasoning-heavy takes more" requirement
- the JSONL log format for critique decisions

`tests/test_pipeline.py` (extended) confirms the same behavior wired
into the real end-to-end pipeline against a real sample contract:
simple questions never call into critique at all, conflict questions
that get an OK verdict cost 2 calls, and conflict questions with a
caught issue cost 3 calls and return the regenerated (not flawed)
answer. All original Level 1 and Level 2 tests continue to pass
unchanged.

---

## Level 4 — Cross-Document Citation Grounding

### Understanding of the goal

Some questions span more than one document (comparing terms, or
reconciling clauses across a set of contracts). Level 4 needs to: pull
relevant evidence out of each document separately, combine those
pieces into one answer, make sure every citation in that answer can be
traced back to a specific retrieved passage (not just "somewhere in
this document"), and check every citation in the final answer against
what was actually retrieved -- recording anything unverifiable as a
warning rather than failing the request or silently dropping it.

### Architecture

```
cross_doc.load_documents(pdf_paths)
  -> [pipeline.load_document_index(path) for each path]   [Level 1, unchanged]
  -> List[LoadedDocument(doc_id, index, full_text, language)]

cross_doc.answer_cross_document(question, documents)
  extract_from_documents(question, documents)
      for doc in documents:
          retrieve.decide(question, doc.doc_id, doc.index, doc.full_text)   [Level 1, unchanged]
          -> search mode: one CrossDocExtraction per retrieved chunk,
             citation_tag = that chunk's own chunk_id (already precise
             to one passage)
          -> full_document mode: one CrossDocExtraction,
             citation_tag = "{doc_id}::full_document"
      -> List[CrossDocExtraction]                          [0 LLM calls -- pure embedding search]

  combined_language = "mixed" if any document is non-English/mixed else "english"
  route.route(combined_language, complexity="cross_document")   [Level 2, +1 new branch]
      -> always tier="strong" (per the assignment's own routing table)

  build_fusion_prompt(question, extractions)   -- every excerpt tagged in [brackets]
  llm.call(fusion_prompt, system_prompt=CROSS_DOC_SYSTEM_PROMPT)   [1 LLM call: the draft]

  critique.answer_with_critique(question, context=<all excerpts>,
                                 draft_answer, is_reasoning_heavy=True, ...)   [Level 3, unchanged]
      -> 0-2 more calls (review, and possibly one regeneration)

  verify_citations(final_answer, extractions)
      -> regex out every [tag], flag any tag not in the retrieved set
      -> warnings, never an exception, never a dropped answer

  -> CrossDocAnswer(answer, doc_ids, extractions, citation_warnings, tier, ...)
```

Total LLM calls per cross-document question: **1 to 3**, completely
independent of how many documents are being compared -- retrieval is
per-document but free (embedding search only); fusion, review, and
regeneration each happen exactly once, no matter whether 2 documents or
20 are involved.

### Key design decisions (and why)

- **Citation tags ARE Level 1's own chunk IDs.** No new citation-tag
  scheme was invented: a chunk's `chunk_id` (e.g.
  `"some_doc::c003"`) already encodes both the document and the exact
  passage, so it doubles as the fusion answer's citation tag directly.
  Verifying a citation is just checking that a bracketed tag exists in
  the set of `citation_tag`s that were actually retrieved for this
  question -- which implicitly verifies the document too, since the
  document id is embedded in the tag. One mechanism covers both "wrong
  passage" and "document not in the collection."
- **Per-document retrieval reuses `retrieve.decide()` completely
  unchanged.** Level 4 doesn't reimplement or duplicate any retrieval,
  chunking, or embedding logic -- it just calls Level 1's own function
  once per document in a loop. This is the same reuse pattern Level
  2/3 already followed for the single-document case.
- **Cross-document questions are *known* to be cross-document by
  construction, not classified from question text.** `cross_doc.py`
  calls `route()` directly with `complexity="cross_document"` rather
  than asking `classify_complexity()` to guess it from the question
  string -- the caller already knows it invoked
  `answer_cross_document()` with multiple documents, so there's nothing
  to infer. This avoids a fragile "does this question span multiple
  documents" heuristic entirely.
- **`route.py` gained exactly one new complexity value and one new
  `elif` branch** (`COMPLEXITY_CROSS_DOCUMENT` -> `TIER_STRONG`),
  directly reflecting the assignment's own routing table, which groups
  "cross-document comparison" with "resolving conflicting/overriding
  clauses" in a single row ("the case Level 3 builds on"). Nothing else
  in `route.py` changed; the existing non-English-overrides-everything
  priority still applies first.
- **Cross-document answers always go through Level 3's critique flow**
  (`is_reasoning_heavy=True`, unconditionally), for the same reason:
  the assignment's routing table explicitly names cross-document
  comparison as one of the two cases Level 3 was built for. This adds
  0-2 calls on top of the 1 fusion call, using the exact same
  `critique.answer_with_critique()` function Level 3 already built --
  no changes to `critique.py` were needed at all; it doesn't care
  whether the "source" text it's checking a draft against came from one
  document or several.
- **Unverifiable citations are warnings on the returned object, not
  exceptions or filtered-out text.** `verify_citations()` never raises
  and never rewrites `answer_text` -- it returns a list of warning
  strings alongside the answer, exactly matching "don't fail the
  request... record it as a warning rather than silently dropping it."
- **A `LoadedDocument` NamedTuple is local to `cross_doc.py`**, wrapping
  `pipeline.load_document_index()`'s existing 5-tuple return value with
  named fields for readability. `pipeline.py` itself was not touched --
  this is purely a convenience type for `cross_doc.py`'s own use.

### Files / functions for this level

- `src/cross_doc.py` (implemented) — `LoadedDocument` (NamedTuple),
  `load_documents(pdf_paths) -> List[LoadedDocument]`,
  `extract_from_documents(question, documents, top_k, threshold) ->
  List[CrossDocExtraction]`, `build_fusion_prompt(question,
  extractions) -> str`, `verify_citations(answer_text, extractions) ->
  List[str]`, `answer_cross_document(question, documents, ...) ->
  CrossDocAnswer`
- `src/models.py` — added `CrossDocExtraction` and `CrossDocAnswer`
  dataclasses, following the existing convention of centralizing all
  domain models in this file
- `src/route.py` — added `COMPLEXITY_CROSS_DOCUMENT` constant and one
  routing branch; everything else in this file is unchanged

### Testing approach

`tests/test_cross_doc.py` (new) covers, with `llm.call` mocked
throughout (no real OpenRouter calls, no API cost):
- per-document extraction pulls evidence from every document, with
  citation tags precise to a real chunk (or a documented
  `::full_document` fallback tag)
- the fusion prompt includes every extraction's tag
- **the Definition of Done's two required cases**: a fully-traceable
  combined answer (zero citation warnings) and a deliberately
  fabricated citation tag that gets flagged as exactly one warning
  without failing the request
- citation verification never raises and deduplicates repeated bad tags
- the full `answer_cross_document()` flow against real sample contracts
  (the NDA and a service agreement), confirming correct doc_ids, tier
  (always `"strong"`), and call counts
- **call count is independent of document count** -- a 3-document
  question costs the same 1-3 calls as a 2-document question
- a flagged issue correctly triggers regeneration, reusing Level 3's
  existing critique flow unchanged
- routing/logging: cross-document decisions log
  `complexity="cross_document"` with a comma-joined `doc_id`

All Level 1, 2, and 3 tests continue to pass unchanged (68 tests
total after this level).
