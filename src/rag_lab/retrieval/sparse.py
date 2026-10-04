"""Sparse retrieval using BM25."""

from rank_bm25 import BM25Okapi

from rag_lab.schemas import Chunk


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer. Good enough for a first pass."""
    return text.lower().split()


class BM25Index:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            self._bm25 = None
            self.chunk_ids = []
            return
        corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        self.chunk_ids = [c.chunk_id for c in chunks]

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self._bm25 is None or not self.chunk_ids:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in order]