from pathlib import Path

from rag_lab.ingest.loaders import load_file


def test_loads_markdown_file(tmp_path: Path):
    raw = tmp_path
    md = raw / "doc.md"
    md.write_text("# My Title\n\nSome content here.", encoding="utf-8")

    doc = load_file(md, raw_root=raw)
    assert doc is not None
    assert doc.title == "My Title"
    assert doc.source_path == "doc.md"
    assert "Some content here" in doc.raw_text
    assert doc.metadata["extension"] == ".md"


def test_loads_txt_file(tmp_path: Path):
    raw = tmp_path
    txt = raw / "notes.txt"
    txt.write_text("First line as title\nSecond line.", encoding="utf-8")

    doc = load_file(txt, raw_root=raw)
    assert doc is not None
    assert doc.title == "First line as title"


def test_unsupported_extension_returns_none(tmp_path: Path):
    raw = tmp_path
    f = raw / "image.png"
    f.write_bytes(b"\x89PNG")
    assert load_file(f, raw_root=raw) is None


def test_empty_file_returns_none(tmp_path: Path):
    raw = tmp_path
    f = raw / "empty.txt"
    f.write_text("   \n\n  ", encoding="utf-8")
    assert load_file(f, raw_root=raw) is None


def test_doc_id_is_stable_across_calls(tmp_path: Path):
    raw = tmp_path
    f = raw / "stable.md"
    f.write_text("# Title\n\nBody", encoding="utf-8")
    a = load_file(f, raw_root=raw)
    b = load_file(f, raw_root=raw)
    assert a is not None and b is not None
    assert a.doc_id == b.doc_id