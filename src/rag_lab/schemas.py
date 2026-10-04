import hashlib
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single source document, before chunking."""

    doc_id: str
    source_path: str
    title: str | None = None
    raw_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_doc_id(source_path: str) -> str:
        """Deterministic ID from the source path."""
        return hashlib.sha256(source_path.encode()).hexdigest()[:16]


class Chunk(BaseModel):
    """A retrievable piece of a document."""

    chunk_id: str
    doc_id: str
    text: str
    position: int
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_chunk_id(doc_id: str, position: int) -> str:
        """Deterministic ID so re-ingestion never duplicates."""
        key = f"{doc_id}:{position}".encode()
        return hashlib.sha256(key).hexdigest()[:16]