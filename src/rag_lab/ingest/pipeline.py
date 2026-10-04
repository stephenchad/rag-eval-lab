"""End-to-end ingest pipeline: walk raw dir, load, chunk, write JSONL."""

from pathlib import Path

from rag_lab.ingest.chunkers import RecursiveChunker
from rag_lab.ingest.loaders import load_file
from rag_lab.schemas import Chunk, Document


def discover_files(raw_dir: Path) -> list[Path]:
    """Return all files under raw_dir, sorted for determinism."""
    return sorted(p for p in raw_dir.rglob("*") if p.is_file())


def ingest_directory(
    raw_dir: Path,
    output_path: Path,
    chunker: RecursiveChunker | None = None,
) -> dict[str, int]:
    """Load every file in raw_dir, chunk it, and write chunks to output_path.

    Returns a summary dict with counts. Overwrites output_path if it exists.
    """
    chunker = chunker or RecursiveChunker()
    raw_dir = raw_dir.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"files_seen": 0, "files_loaded": 0, "files_skipped": 0, "chunks_written": 0}

    with output_path.open("w", encoding="utf-8") as f:
        for path in discover_files(raw_dir):
            stats["files_seen"] += 1
            doc: Document | None = load_file(path, raw_root=raw_dir)
            if doc is None:
                stats["files_skipped"] += 1
                continue

            stats["files_loaded"] += 1
            chunks: list[Chunk] = chunker.chunk(doc)
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")
                stats["chunks_written"] += 1

    return stats


def read_chunks(path: Path) -> list[Chunk]:
    """Read a chunks.jsonl file back into Chunk objects."""
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk.model_validate_json(line))
    return chunks