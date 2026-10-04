"""Information retrieval metrics.

All functions take a ranked list of retrieved doc ids and a single ground-truth
doc id, and return a float in [0, 1] unless otherwise noted.
"""


def hit_at_k(ranked: list[str], expected: str, k: int) -> float:
    """1.0 if the expected doc is in the top k, else 0.0."""
    return 1.0 if expected in ranked[:k] else 0.0


def precision_at_k(ranked: list[str], expected: str, k: int) -> float:
    """Fraction of the top k results that match the expected doc."""
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r == expected) / len(top)


def recall_at_k(ranked: list[str], expected: str, k: int) -> float:
    """1.0 if the expected doc appears in the top k, else 0.0.

    With a single relevant document per query, recall@k equals hit@k.
    Kept separate for clarity and future multi-relevant setups.
    """
    return hit_at_k(ranked, expected, k)


def reciprocal_rank(ranked: list[str], expected: str) -> float:
    """1 / rank of the first correct result, or 0 if it never appears."""
    for i, r in enumerate(ranked, start=1):
        if r == expected:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    ranked_lists: list[list[str]], expected_docs: list[str]
) -> float:
    """MRR across a set of queries."""
    if not ranked_lists:
        return 0.0
    rrs = [reciprocal_rank(r, e) for r, e in zip(ranked_lists, expected_docs)]
    return sum(rrs) / len(rrs)


def aggregate_at_k(
    ranked_lists: list[list[str]], expected_docs: list[str], k: int
) -> dict[str, float]:
    """Convenience aggregate for a batch of queries at a given k."""
    n = len(ranked_lists)
    if n == 0:
        return {"hit@k": 0.0, "precision@k": 0.0, "recall@k": 0.0, "mrr": 0.0}
    return {
        f"hit@{k}": sum(hit_at_k(r, e, k) for r, e in zip(ranked_lists, expected_docs)) / n,
        f"precision@{k}": sum(
            precision_at_k(r, e, k) for r, e in zip(ranked_lists, expected_docs)
        ) / n,
        f"recall@{k}": sum(
            recall_at_k(r, e, k) for r, e in zip(ranked_lists, expected_docs)
        ) / n,
        "mrr": mean_reciprocal_rank(ranked_lists, expected_docs),
    }