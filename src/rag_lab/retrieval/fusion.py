"""Reciprocal Rank Fusion.

Combines multiple ranked lists into a single ranking.
Reference: Cormack et al. 2009, "Reciprocal Rank Fusion outperforms Condorcet".
"""


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked lists of chunk ids. Higher score = better."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])