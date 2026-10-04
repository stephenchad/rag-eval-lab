# rag-eval-lab

A retrieval-augmented generation (RAG) system built in phases, with an emphasis on **measurable retrieval quality** rather than demo gloss.

The project currently ships a **hybrid retriever** (dense + sparse, fused with Reciprocal Rank Fusion) over a small corpus, evaluated by a **CI-gated harness** with hand-labeled questions.

## Retrieval quality (current)

| Metric | Value | Notes |
|--------|-------|-------|
| hit@3 | **1.000** | Expected doc retrieved in top 3 for all 15 questions |
| recall@3 | **1.000** | Same value as hit@3 (single relevant doc per query) |
| MRR | **0.933** | Correct doc usually ranked 1st |
| precision@3 | 0.333 | Ceiling is 1/3 — each query has only one correct doc |

**Honest caveat:** the corpus is small (3 documents) and the questions were hand-written against it. hit@3 = 1.0 should be read as "the plumbing works and the corpus is small," not "the retriever is state of the art." The eval harness is designed to be re-run against larger corpora where these numbers would be meaningful.

## Architecture

```
data/raw/*.md,*.txt
    │
    ▼
[ingest.loaders]     →  Document objects (stable doc_id from path)
    │
    ▼
[ingest.chunkers]    →  RecursiveChunker (paragraph → line → sentence → word)
    │
    ▼
data/processed/chunks.jsonl
    │
    ▼
[retrieval.hybrid]   ┌─ DenseIndex: fastembed (bge-small-en-v1.5) + numpy cosine
                     └─ BM25Index: rank_bm25 keyword search
                            │
                            ▼
                     Reciprocal Rank Fusion
    │
    ▼
Top-k chunks with source attribution
```

## Evaluation

The eval harness is the point of this project. It runs 15 hand-labeled questions against the retriever and gates CI on two thresholds:

- `hit@3 >= 0.80`
- `MRR >= 0.65`

If a refactor regresses retrieval quality, `pytest` fails. Run it manually:

```
python scripts\eval.py
```

## Build phases

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | Ingestion and chunking pipeline | ✅ complete |
| 2 | Hybrid retrieval (dense + BM25 + RRF) | ✅ complete |
| 3 | Grounded generation with citations | ⏸ deferred — no LLM budget |
| 4 | REST API and containerization | ⏸ deferred — no Docker on dev machine |
| 5 | Evaluation harness with CI gate | ✅ complete |
| 6 | Tracing and observability | ⬜ optional next |
| 7 | Case-study README | 🔄 this document |

### What was deliberately left out, and why

- **Grounded generation** — the generation layer requires an LLM. Given the constraints of this build (no API budget, no local GPU), retrieval and evaluation were prioritized. Retrieval is also the part that benefits most from rigorous measurement.
- **Vector database** — at three documents, brute-force numpy cosine over a `(N, 384)` matrix is instantaneous and simpler than running Qdrant. The switch point is documented: when N grows past roughly 10⁵ chunks, approximate nearest neighbor indexing becomes worthwhile.
- **Docker** — the development machine doesn't support Docker. Local `python scripts\ingest.py` and `pytest` are sufficient for a project of this size. Containerization would be a natural next step for deployment.

## Running it locally

Requires Python 3.11+.

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# Put .md or .txt files in data/raw/, then:
python scripts\ingest.py

# Run the eval harness:
python scripts\eval.py

# Run the test suite (includes the eval gate):
pytest -v
```

## Stack

- **Parsing / chunking:** custom recursive character chunker (`src/rag_lab/ingest/`)
- **Embeddings:** `fastembed` with `BAAI/bge-small-en-v1.5` (ONNX, ~30 MB model)
- **Sparse retrieval:** `rank_bm25`
- **Fusion:** Reciprocal Rank Fusion (Cormack et al., 2009)
- **Validation:** `pydantic` v2 models
- **Tests:** `pytest` — 36 tests, including the retrieval quality gate
- **Lint:** `ruff`