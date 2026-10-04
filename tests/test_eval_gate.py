"""End-to-end retrieval quality gate.

This runs the actual HybridRetriever over the real corpus and asserts that
quality does not regress below fixed thresholds. It is the guardrail that
keeps a refactor from silently breaking retrieval.
"""

from pathlib import Path

import pytest

from rag_lab.config import settings
from rag_lab.eval.harness import load_cases, run_eval
from rag_lab.ingest.pipeline import read_chunks
from rag_lab.retrieval.hybrid import HybridRetriever

# Minimum acceptable quality. Tune these once and treat them as a floor.
MIN_HIT_AT_3 = 0.80
MIN_MRR = 0.65


@pytest.fixture(scope="module")
def eval_result():
    chunks_path = settings.processed_dir / "chunks.jsonl"
    if not chunks_path.exists():
        pytest.skip("run `python scripts\\ingest.py` first to build the corpus")

    questions_path = Path("data/eval/questions.jsonl")
    if not questions_path.exists():
        pytest.skip("missing data/eval/questions.jsonl")

    chunks = read_chunks(chunks_path)
    retriever = HybridRetriever()
    retriever.build(chunks)

    cases = load_cases(questions_path)
    return run_eval(cases, retriever, k=3)


def test_hit_at_3_above_threshold(eval_result):
    score = eval_result.metrics["hit@3"]
    assert score >= MIN_HIT_AT_3, (
        f"hit@3 = {score:.3f} is below the floor of {MIN_HIT_AT_3}. "
        f"Failing cases: {[c.qid for c in eval_result.failing_cases(3)]}"
    )


def test_mrr_above_threshold(eval_result):
    score = eval_result.metrics["mrr"]
    assert score >= MIN_MRR, f"MRR = {score:.3f} is below the floor of {MIN_MRR}"