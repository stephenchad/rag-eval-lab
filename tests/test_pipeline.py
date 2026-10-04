import json
from pathlib import Path

from rag_lab.ingest.chunkers import RecursiveChunker
from rag_lab.ingest.pipeline import ingest_directory, read_chunks


def _write_fixture_files(raw: Path) -> None:
    (raw / "a.md").write_text(
        "# Doc A\n\n" + "word " * 300, encoding="utf-8"
    )
    (raw / "b.txt").write_text(
        "Doc B\n\n" + "sentence. " * 200, encoding="utf-8"
    )
    (raw / "skip.bin").write_bytes(b"\x00\x01\x02")


def test_ingest_writes_valid_jsonl(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_fixture_files(raw)
    out = tmp_path / "processed" / "chunks.jsonl"

    stats = ingest_directory(raw, out, chunker=RecursiveChunker(target_tokens=80))

    assert out.exists()
    assert stats["files_loaded"] == 2
    assert stats["files_skipped"] == 1
    assert stats["chunks_written"] == stats["files_loaded"] + 0 or stats["chunks_written"] > 0

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == stats["chunks_written"]
    for line in lines:
        obj = json.loads(line)
        assert "chunk_id" in obj
        assert "text" in obj
        assert obj["text"]


def test_ingest_is_idempotent(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_fixture_files(raw)
    out = tmp_path / "processed" / "chunks.jsonl"

    ingest_directory(raw, out, chunker=RecursiveChunker(target_tokens=80))
    first = read_chunks(out)

    ingest_directory(raw, out, chunker=RecursiveChunker(target_tokens=80))
    second = read_chunks(out)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_read_chunks_roundtrip(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "x.md").write_text("# X\n\n" + "content " * 100, encoding="utf-8")
    out = tmp_path / "processed" / "chunks.jsonl"

    stats = ingest_directory(raw, out, chunker=RecursiveChunker(target_tokens=60))
    chunks = read_chunks(out)

    assert len(chunks) == stats["chunks_written"]
    for c in chunks:
        assert c.text
        assert c.chunk_id
        assert c.doc_id