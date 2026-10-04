"""CLI: run the ingest pipeline over data/raw and write data/processed/chunks.jsonl."""

from rag_lab.config import settings
from rag_lab.ingest.pipeline import ingest_directory


def main() -> None:
    settings.ensure_dirs()
    output = settings.processed_dir / "chunks.jsonl"
    stats = ingest_directory(settings.raw_dir, output)
    print(f"Wrote {stats['chunks_written']} chunks to {output}")
    print(
        f"Files seen: {stats['files_seen']}, "
        f"loaded: {stats['files_loaded']}, "
        f"skipped: {stats['files_skipped']}"
    )


if __name__ == "__main__":
    main()