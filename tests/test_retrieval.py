from rag_lab.retrieval.fusion import reciprocal_rank_fusion
from rag_lab.retrieval.sparse import BM25Index
from rag_lab.schemas import Chunk


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="d1", text=text, position=0, token_count=len(text.split())
    )


def test_rrf_prefers_items_in_both_lists():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    fused = reciprocal_rank_fusion([a, b])
    ids = [cid for cid, _ in fused]
    # x and y appear in both lists and should outrank z and w
    assert set(ids[:2]) == {"x", "y"}


def test_bm25_finds_keyword_match():
    chunks = [
        _chunk("c1", "the transformer uses multi-head attention"),
        _chunk("c2", "asyncio is for concurrent io in python"),
        _chunk("c3", "vector search finds similar embeddings"),
    ]
    index = BM25Index()
    index.build(chunks)
    hits = index.search("multi-head attention", top_k=1)
    assert hits[0][0] == "c1"


def test_bm25_empty_index_returns_empty():
    index = BM25Index()
    index.build([])
    assert index.search("anything") == []