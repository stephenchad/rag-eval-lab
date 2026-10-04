"""Hybrid retriever combining dense and sparse search via RRF."""

from rag_lab.retrieval.dense import DenseIndex
from rag_lab.retrieval.fusion import reciprocal_rank_fusion
from rag_lab.retrieval.sparse import BM25Index
from rag_lab.schemas import Chunk


class HybridRetriever:
    def __init__(self) -> None:
        self.dense = DenseIndex()
        self.sparse = BM25Index()
        self._chunks_by_id: dict[str, Chunk] = {}

    def build(self, chunks: list[Chunk]) -> None:
        self.dense.build(chunks)
        self.sparse.build(chunks)
        self._chunks_by_id = {c.chunk_id: c for c in chunks}

    def retrieve(self, query: str, top_k: int = 3) -> list[Chunk]:
        dense_hits = [cid for cid, _ in self.dense.search(query, top_k=10)]
        sparse_hits = [cid for cid, _ in self.sparse.search(query, top_k=10)]
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits])
        return [
            self._chunks_by_id[cid]
            for cid, _ in fused[:top_k]
            if cid in self._chunks_by_id
        ]