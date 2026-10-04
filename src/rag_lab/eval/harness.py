"""Runs a retriever against a labeled question set and reports metrics."""

import json
from dataclasses import dataclass
from pathlib import Path

from rag_lab.eval.metrics import aggregate_at_k
from rag_lab.retrieval.hybrid import HybridRetriever


@dataclass
class EvalCase:
    qid: str
    question: str
    expected_doc: str


@dataclass
class EvalResult:
    cases: list[EvalCase]
    ranked_docs_per_case: list[list[str]]
    metrics: dict[str, float]

    def failing_cases(self, k: int) -> list[EvalCase]:
        """Cases where the expected doc was not in the top k."""
        return [
            case
            for case, ranked in zip(self.cases, self.ranked_docs_per_case)
            if case.expected_doc not in ranked[:k]
        ]


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(
                EvalCase(
                    qid=obj["qid"],
                    question=obj["question"],
                    expected_doc=obj["expected_doc"],
                )
            )
    return cases


def run_eval(
    cases: list[EvalCase],
    retriever: HybridRetriever,
    k: int = 3,
) -> EvalResult:
    ranked_per_case: list[list[str]] = []
    for case in cases:
        hits = retriever.retrieve(case.question, top_k=k)
        # Deduplicate docs while preserving rank order.
        seen: set[str] = set()
        ranked_docs: list[str] = []
        for hit in hits:
            doc = hit.metadata.get("source_path", "")
            if doc and doc not in seen:
                seen.add(doc)
                ranked_docs.append(doc)
        ranked_per_case.append(ranked_docs)

    expected = [c.expected_doc for c in cases]
    metrics = aggregate_at_k(ranked_per_case, expected, k)
    return EvalResult(cases=cases, ranked_docs_per_case=ranked_per_case, metrics=metrics)