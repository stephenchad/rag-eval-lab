"""Recursive character chunking.

Splits documents into chunks by trying the largest structural separator first
(paragraph breaks), then falling back to smaller ones (lines, sentences, words)
when a piece is still too big. Adjacent chunks overlap so an answer spanning a
boundary is still fully contained in at least one chunk.
"""

from typing import Protocol

from rag_lab.config import settings
from rag_lab.schemas import Chunk, Document

# Separators ordered from largest structural unit to smallest.
SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

# Rough characters-per-token when we need to hard-split.
_CHARS_PER_TOKEN = 5


def approx_tokens(text: str) -> int:
    """Cheap token approximation. Words are close enough for chunk sizing."""
    return len(text.split())


class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...


class RecursiveChunker:
    def __init__(
        self,
        target_tokens: int | None = None,
        overlap_tokens: int | None = None,
        min_tokens: int | None = None,
    ) -> None:
        self.target_tokens = target_tokens or settings.chunk_target_tokens
        self.overlap_tokens = (
            overlap_tokens if overlap_tokens is not None else 50
        )
        self.min_tokens = min_tokens or settings.chunk_min_tokens

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.raw_text.strip()
        if not text:
            return []

        fragments = self._split(text)
        packed = self._pack(fragments)
        overlapped = self._apply_overlap(packed)
        cleaned = self._cleanup(overlapped)

        if not cleaned:
            return []

        return self._to_chunks(doc, cleaned)

    # ------------------------------------------------------------------ helpers

    def _split(self, text: str, sep_index: int = 0) -> list[str]:
        """Recursively split text, preferring larger separators."""
        if approx_tokens(text) <= self.target_tokens:
            return [text]
        if sep_index >= len(SEPARATORS):
            return [text]

        sep = SEPARATORS[sep_index]

        if sep == "":
            # Last resort: hard split by characters.
            step = max(1, self.target_tokens * _CHARS_PER_TOKEN)
            pieces = [text[i : i + step] for i in range(0, len(text), step)]
        else:
            pieces = text.split(sep)

        out: list[str] = []
        for piece in pieces:
            if not piece.strip():
                continue
            if approx_tokens(piece) > self.target_tokens:
                out.extend(self._split(piece, sep_index + 1))
            else:
                out.append(piece)
        return out

    def _pack(self, fragments: list[str]) -> list[str]:
        """Greedily pack fragments into chunks no larger than target_tokens."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for frag in fragments:
            frag_tokens = approx_tokens(frag)
            if current and current_tokens + frag_tokens > self.target_tokens:
                chunks.append(" ".join(current))
                current = [frag]
                current_tokens = frag_tokens
            else:
                current.append(frag)
                current_tokens += frag_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Prepend the tail of each previous chunk to the next one."""
        if self.overlap_tokens <= 0 or len(chunks) <= 1:
            return chunks

        out: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1].split()
            tail = prev_words[-self.overlap_tokens :]
            if tail:
                out.append(" ".join(tail) + " " + chunks[i])
            else:
                out.append(chunks[i])
        return out

    def _cleanup(self, chunks: list[str]) -> list[str]:
        """Merge very small chunks into neighbors when it doesn't blow the budget."""
        if len(chunks) <= 1:
            return chunks

        cleaned: list[str] = []
        for chunk in chunks:
            if (
                cleaned
                and approx_tokens(chunk) < self.min_tokens
                and approx_tokens(cleaned[-1]) + approx_tokens(chunk)
                <= self.target_tokens
            ):
                cleaned[-1] = cleaned[-1] + " " + chunk
            else:
                cleaned.append(chunk)

        # If the last chunk is too small and can't be merged forward, drop it.
        if (
            len(cleaned) > 1
            and approx_tokens(cleaned[-1]) < self.min_tokens
        ):
            cleaned.pop()

        return cleaned

    def _to_chunks(self, doc: Document, texts: list[str]) -> list[Chunk]:
        """Convert text pieces into Chunk objects with propagated metadata."""
        metadata = {
            "source_path": doc.source_path,
            "title": doc.title,
            **doc.metadata,
        }
        out: list[Chunk] = []
        for i, text in enumerate(texts):
            stripped = text.strip()
            if not stripped:
                continue
            out.append(
                Chunk(
                    chunk_id=Chunk.make_chunk_id(doc.doc_id, i),
                    doc_id=doc.doc_id,
                    text=stripped,
                    position=i,
                    token_count=approx_tokens(stripped),
                    metadata=dict(metadata),
                )
            )
        return out