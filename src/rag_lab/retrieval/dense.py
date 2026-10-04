"""Dense retrieval using fastembed + numpy cosine similarity.

Vectors are stored as a .npy matrix alongside a parallel list of chunk ids,
so we can map a row index back to the chunk it belongs to.
"""

from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from rag_lab.schemas import Chunk

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class DenseIndex:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: TextEmbedding | None = None
        self.vectors: np.ndarray | None = None
        self.chunk_ids: list[str] = []

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def build(self, chunks: list[Chunk]) -> None:
        """Embed every chunk and store the normalized vector matrix."""
        if not chunks:
            self.vectors = np.zeros((0, 0), dtype=np.float32)
            self.chunk_ids = []
            return

        texts = [c.text for c in chunks]
        vecs = np.array(list(self.model.embed(texts)), dtype=np.float32)
        # L2-normalize so cosine similarity is just a dot product.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = vecs / norms
        self.chunk_ids = [c.chunk_id for c in chunks]

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return (chunk_id, score) sorted by descending similarity."""
        if self.vectors is None or len(self.chunk_ids) == 0:
            return []
        q = np.array(list(self.model.embed([query])), dtype=np.float32)[0]
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm
        scores = self.vectors @ q
        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idx]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.vectors is None:
            raise RuntimeError("call build() before save()")
        np.save(path.with_suffix(".npy"), self.vectors)
        path.with_suffix(".ids.json").write_text(
            __import__("json").dumps(self.chunk_ids), encoding="utf-8"
        )

    def load(self, path: Path) -> None:
        self.vectors = np.load(path.with_suffix(".npy"))
        import json

        self.chunk_ids = json.loads(
            path.with_suffix(".ids.json").read_text(encoding="utf-8")
        )