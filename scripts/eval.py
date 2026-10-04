"""CLI: run the retrieval eval and print a metrics table."""

from pathlib import Path

from rag_lab.config import settings
from rag_lab.eval.harness import load_cases, run_eval
from rag_lab.ingest.pipeline import read_chunks
from rag_lab.retrieval.hybrid import HybridRetriever


def main() -> None:
    chunks_path = settings.processed_dir / "chunks.jsonl"
    questions_path = Path("data/eval/questions.jsonl")

    if not chunks_path.exists():
        print(f"No chunks found at {chunks_path}. Run: python scripts\\ingest.py")
        return

    chunks = read_chunks(chunks_path)
    retriever = HybridRetriever()
    retriever.build(chunks)

    cases = load_cases(questions_path)
    result = run_eval(cases, retriever, k=3)

    print(f"\nRan {len(cases)} questions, top_k=3\n")
    print("Metrics:")
    for name, value in result.metrics.items():
        print(f"  {name:<14} {value:.3f}")

    failing = result.failing_cases(3)
    if failing:
        print(f"\nFailing cases ({len(failing)}):")
        for case in failing:
            print(f"  {case.qid}: {case.question}")
            print(f"        expected: {case.expected_doc}")
    else:
        print("\nNo failing cases.")


if __name__ == "__main__":
    main()