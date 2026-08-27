# MikeLegal RAG Assistant

A retrieval-augmented question-answering system for legal contract PDFs. It grounds answers in retrieved document evidence, routes questions to different model tiers based on language and complexity, critiques and revises reasoning-heavy answers, and can answer questions that span multiple documents with traceable, verified citations.

## Overview

Legal documents raise two problems for a QA system: answers need to be grounded in the actual document text rather than invented, and different questions call for very different amounts of work — a casual greeting doesn't need the same handling as a question about which of two conflicting clauses takes precedence. This project is organized around that idea: each capability layer (retrieval, routing, critique, cross-document fusion) only spends extra computation when the question actually needs it.

## Key Features

- **Grounded single-document QA** — answers a question against one contract, retrieving relevant passages or falling back to the full document when retrieval isn't confident.
- **Rule-based model routing** — picks one of three model tiers per question using cheap signals (document language, question complexity), with no extra LLM call spent on the routing decision itself.
- **Critique and regeneration** — an independent reviewer pass checks reasoning-heavy answers (conflicting/overriding clauses) and triggers a single regeneration attempt if it finds a real issue.
- **Cross-document QA with citation grounding** — answers questions across multiple documents in a single fusion call, tags every claim with a traceable citation, and verifies each citation against what was actually retrieved.
- **Full decision logging** — every retrieval, routing, and critique decision is appended as a JSON line to `logs/run_log.jsonl`, so each question's cost and reasoning are auditable.

## Architecture

```
Document PDF
   │
   ▼
extraction (src/extract.py)        page-by-page text extraction
   │
   ▼
chunking (src/chunk.py)            clause-level passages
   │
   ▼
embeddings/index (src/index.py)    in-memory embedding index
   │
   ▼
retrieval (src/retrieve.py)        search vs. full-document decision
   │
   ▼
routing (src/route.py)             language + complexity → model tier
   │
   ▼
LLM (src/llm.py)                   draft answer
   │
   ▼
critique/revision (src/critique.py)   review + optional regeneration
   │
   ▼
grounded answer
```

For questions spanning multiple documents, `src/cross_doc.py` runs retrieval independently per document, fuses the tagged excerpts into a single prompt, and routes the combined answer through the same routing and critique logic as a single-document question:

```
Document A ─┐
             ├─ per-document retrieval → tagged excerpts → fusion prompt → LLM → critique → answer + citation check
Document B ─┘
```

All single-document steps are wired together in `src/pipeline.py`.

## Capabilities

The system is organized into layered capabilities, each building on the previous one without changing it.

| Capability | Problem it solves | Triggered when | LLM calls |
|---|---|---|---|
| Single-document QA | Answer a question grounded in one contract | Any question about a loaded document | 1 |
| Model routing | Match model strength to question difficulty | Every question, before the draft is generated | 0 (routing itself is rule-based) |
| Critique & revision | Catch unaddressed conflicts/carve-outs in reasoning-heavy answers | Only for questions classified `conflict_or_override` | +1 (review), +1 more only if an issue is found |
| Cross-document QA | Answer questions that require comparing multiple documents | Questions involving more than one document | 1 (fusion) + 0–2 (critique) |

Casual, document-independent questions skip retrieval entirely and cost exactly one LLM call. Simple factual and full-document-reasoning questions also cost one call. Only `conflict_or_override` questions and cross-document questions can trigger the additional critique calls.

## Retrieval

Each document is chunked into clause-level passages and embedded into an in-memory index (`EmbeddedIndex`). For a given question:

- The index is searched for the `top_k` most similar chunks.
- If the best match's similarity score clears a configurable `threshold`, the pipeline answers from the retrieved passages (`mode: search`).
- If no chunk clears the threshold, the pipeline falls back to reading the entire document instead of returning a low-confidence match (`mode: full_document`).

`top_k` and `threshold` are plain parameters on `answer_single_document(...)` / `retrieve.decide(...)` — they can be set per call with no config file or restart required:

```python
answer_single_document(question, pdf_path=path, top_k=5, threshold=0.5)
```

The fallback exists so that a question the retriever genuinely can't find good evidence for still gets answered from the full document rather than from a weak, potentially misleading match.

## Model Routing

Each question is routed to one of three model tiers — `light`, `standard`, or `strong` — using two rule-based signals, with no LLM call spent on the decision:

- **Document language** (`src/route.py: detect_language`) — detected once per document (inside `load_document_index`) and reused for every question about that document.
- **Question complexity** (`src/route.py: classify_complexity`) — classified per question as `casual`, `simple_factual`, `full_document_reasoning`, or `conflict_or_override`, reusing retrieval's own `mode` as one of the signals.

`route()` combines both signals into a tier plus a short reason:

- A non-English or mixed-language document always routes to the strongest tier, regardless of complexity.
- Otherwise, the tier follows question complexity.
- An unrecognized case safely falls back to the `standard` tier.

`llm.resolve_model_for_tier(tier)` maps a tier name to a real OpenRouter model id, read from `MODEL_TIER_LIGHT` / `MODEL_TIER_STANDARD` / `MODEL_TIER_STRONG` in `.env` (the same model family is used for all three tiers).

Every routing decision is logged to `logs/run_log.jsonl` (`"event": "routing"`) alongside the corresponding retrieval log line, so both decisions for a question can be inspected together.

## Critique and Revision

For questions classified `conflict_or_override` (resolving conflicting or overriding clauses), one independent review pass runs after the draft answer. Every other question type is untouched and still costs exactly one LLM call.

- **`review()`** — a single call, using a distinct reviewer system prompt, checks the draft against the source text for an unaddressed carve-out/override, an unsupported claim, or an unreconciled contradiction between clauses. Returns `VERDICT: OK` or `VERDICT: ISSUE` with an explanation.
- **`regenerate()`** — only called if review flags a real issue. One regeneration attempt folds in the reviewer's feedback. If the issue still can't be resolved, the response is prefixed with `"UNRESOLVED:"` so the caller can detect it without a fourth call; the original draft is returned with the discrepancy recorded rather than dropped silently.
- **`answer_with_critique()`** — orchestrates the above and reports the exact number of calls used:

| Case | Calls |
|---|---|
| Not reasoning-heavy | 1 (draft only) |
| Reasoning-heavy, review says OK | 2 |
| Reasoning-heavy, issue found and regenerated | 3 |

The reviewer and regenerator reuse the same strong-tier model the draft already used, but in a genuinely separate role — a distinct system prompt that only sees the draft's output, not its reasoning. Every critique decision is logged (`"event": "critique"`), including the 1-call case, so the added cost per question is directly measurable.

## Cross-Document Question Answering

Questions that require comparing or reconciling terms across multiple documents are handled by `src/cross_doc.py`:

- **`load_documents(pdf_paths)`** — loads and embeds each document once, reusing `load_document_index` unchanged, so a set of documents can be loaded once and reused across many cross-document questions.
- **`extract_from_documents(question, documents)`** — runs retrieval's own `decide()` once per document (pure embedding search, no LLM calls). A confident match cites the retrieved chunk's own `chunk_id` (e.g. `some_doc::c003`); a full-document fallback cites `"{doc_id}::full_document"` instead.
- **`answer_cross_document(question, documents)`** — builds one fusion prompt from all tagged excerpts and makes exactly one LLM call for a combined answer. Cross-document comparison always routes to the strong tier, then passes through the same critique flow used for single-document `conflict_or_override` questions — adding 0–2 more calls for review and a possible regeneration.

**Total cost: 1–3 LLM calls, regardless of how many documents are being compared.**

## Citation Grounding

Every citation tag in a cross-document answer refers to a specific piece of retrieved evidence — either an individual chunk (`doc_id::chunk_id`) or a full-document fallback (`doc_id::full_document`).

`verify_citations(answer, extractions)` checks every bracketed citation in the final answer against what was actually retrieved. An unverifiable citation (referring to a document or passage that wasn't actually retrieved) is recorded as a warning on the result — the request still succeeds. Warnings are preferred over silently dropping or failing the request, because a flagged, human-checkable citation is more useful than one that looks trustworthy but isn't verifiable.

## Project Structure

```
data/
  contracts/          sample contract PDFs used for examples and tests
logs/
  run_log.jsonl        JSON-lines log of retrieval, routing, and critique decisions
src/
  extract.py            page-by-page PDF text extraction
  chunk.py               clause-level passage splitting
  index.py                embedding index (EmbeddedIndex)
  retrieve.py            search vs. full-document decision logic
  route.py                language detection and complexity classification
  llm.py                  LLM calls and tier→model resolution (OpenRouter)
  critique.py             review and regeneration
  cross_doc.py            multi-document loading, fusion, and citation verification
  pipeline.py             wires extraction → chunking → indexing → retrieval → routing → LLM → critique
tests/
  conftest.py             deterministic fake encoder for offline retrieval tests
  test_route.py           routing tests
  test_critique.py        critique/regeneration tests
  test_cross_doc.py        cross-document tests
  test_pipeline.py        end-to-end pipeline tests
.env.example
.gitignore
PLAN.md
README.md
REFLECTIONS.md
requirements.txt
```

## Installation

```bash
git clone <repository-url>
cd mikelegal-rag-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set a real `OPENROUTER_API_KEY`, plus the three routing tier variables:

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Authenticates LLM calls via OpenRouter |
| `MODEL_TIER_LIGHT` | OpenRouter model id used for the `light` tier |
| `MODEL_TIER_STANDARD` | OpenRouter model id used for the `standard` tier |
| `MODEL_TIER_STRONG` | OpenRouter model id used for the `strong` tier |

On first use, `EmbeddedIndex` downloads the `all-MiniLM-L6-v2` encoder from Hugging Face (a few hundred MB, cached locally afterward). This requires network access to `huggingface.co`.

## Usage

**Single-document question:**

```python
from src.pipeline import answer_single_document

result = answer_single_document(
    question="What is the notice period for termination?",
    pdf_path="data/contracts/NON disclosure agreement Edited.pdf",
)
print(result.mode, result.tier, result.confidence, result.answer)
```

**Reusing an index across multiple questions on the same document:**

```python
from src.pipeline import load_document_index, answer_single_document

doc_id, chunks, index, full_text = load_document_index(
    "data/contracts/NON disclosure agreement Edited.pdf"
)

for question in ["Who are the parties?", "What happens on breach?"]:
    result = answer_single_document(
        question, doc_id=doc_id, index=index, full_text=full_text
    )
    print(question, "->", result.mode, "->", result.answer)
```

**Casual questions (no retrieval):**

```python
from src.pipeline import answer_casual

result = answer_casual("hey, how are you?")
print(result.tier, result.answer)
```

**Cross-document comparison:**

```python
from src.cross_doc import load_documents, answer_cross_document

docs = load_documents([
    "data/contracts/NON disclosure agreement Edited.pdf",
    "data/contracts/Service agreement new clm Copy.pdf",
])
result = answer_cross_document(
    "Compare the confidentiality terms across these two agreements.",
    docs,
)
print(result.tier, result.n_llm_calls, result.citation_warnings, result.answer)
```

Load documents once with `load_documents` and reuse the returned `docs` for further cross-document questions — no re-extraction or re-embedding.

There is currently no standalone CLI or web application entry point; the project is used through the functions in `src/pipeline.py` and `src/cross_doc.py`.

## Testing

```bash
pytest tests/ -v
```

**68 tests passing.**

Tests never make real OpenRouter calls (the LLM is mocked) and never require the real embedding model or network access — retrieval-logic tests use a small deterministic fake encoder (`tests/conftest.py`). Chunking and extraction tests run against real sample contracts in `data/contracts/`.

Coverage includes:

- Extraction/chunking metadata and deterministic chunk IDs
- Search vs. full-document fallback retrieval modes, with configurable `top_k`/`threshold`
- JSONL logging fields for retrieval, routing, and critique events
- Language detection, complexity classification, all routing cases, and tier→model resolution
- Critique: a planted-conflict scenario that must be caught and reconciled, and a companion case where regeneration still can't resolve it
- Cross-document: a fully traceable combined answer with zero citation warnings, and a deliberately fabricated citation that is flagged without failing the request
- End-to-end pipeline tests with a mocked LLM, including routed model ids reaching `llm.call()`

## Design Decisions

- **Retrieval with full-document fallback** — a similarity threshold prevents the system from confidently answering off a weak match; falling back to the full document trades some efficiency for correctness when retrieval is uncertain.
- **Rule-based routing instead of LLM-based routing** — language and complexity are cheap, deterministic signals, so routing costs nothing in LLM calls and stays fast and predictable.
- **Critique limited to reasoning-heavy questions** — review and regeneration are expensive relative to a single draft call, so they're reserved for the question type where an unreconciled conflict is a real correctness risk.
- **Document/index reuse** — extraction and embedding are done once per document and reused across questions and across cross-document comparisons, avoiding redundant work.
- **Citation grounding via verification, not enforcement** — verifying citations after generation, rather than constraining generation itself, keeps the fusion prompt simple while still surfacing untrustworthy citations as warnings.
- **Cross-document fusion as a single LLM call** — building one fusion prompt from all tagged excerpts keeps cross-document cost bounded (1–3 calls total) regardless of how many documents are involved.

## Limitations

- Input is assumed to be text-extractable contract PDFs; scanned/image-only PDFs are not handled.
- Requires a configured OpenRouter API key and network access for LLM calls, and (on first run) network access to Hugging Face to download the embedding model.
- Retrieval quality depends on the chunking granularity and the embedding model; the full-document fallback mitigates but does not eliminate low-recall cases.
- Citation verification checks that a citation refers to something that was actually retrieved — it does not independently verify that the LLM's claim is fully supported by that passage's content.
- No standalone CLI or web interface is currently provided.