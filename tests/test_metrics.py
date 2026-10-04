from rag_lab.eval.metrics import (
    aggregate_at_k,
    hit_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    reciprocal_rank,
)


def test_hit_at_k_true():
    assert hit_at_k(["a", "b", "c"], "b", k=3) == 1.0


def test_hit_at_k_false_when_out_of_range():
    assert hit_at_k(["a", "b", "c"], "b", k=1) == 0.0


def test_precision_at_k_perfect():
    assert precision_at_k(["a", "a", "a"], "a", k=3) == 1.0


def test_precision_at_k_partial():
    assert precision_at_k(["a", "b", "a"], "a", k=3) == 2 / 3


def test_reciprocal_rank_first():
    assert reciprocal_rank(["a", "b"], "a") == 1.0


def test_reciprocal_rank_second():
    assert reciprocal_rank(["a", "b"], "b") == 0.5


def test_reciprocal_rank_missing():
    assert reciprocal_rank(["a", "b"], "z") == 0.0


def test_mrr_average():
    ranked = [["a", "b"], ["b", "a"], ["c", "b", "a"]]
    expected = ["a", "b", "a"]
    # RRs: 1.0, 1.0, 1/3
    assert abs(mean_reciprocal_rank(ranked, expected) - (1 + 1 + 1 / 3) / 3) < 1e-9


def test_aggregate_at_k_shape():
    ranked = [["a", "b"], ["b", "a"]]
    expected = ["a", "b"]
    result = aggregate_at_k(ranked, expected, k=2)
    assert "hit@2" in result
    assert "precision@2" in result
    assert "recall@2" in result
    assert "mrr" in result
    assert result["hit@2"] == 1.0
    assert result["mrr"] == 1.0